"""
DPO Trainer for diffusion models.

Handles the full training loop:
  1. Load preference pairs (winner/loser).
  2. Encode to latent space.
  3. Add noise at random timesteps.
  4. Compute log-probs under policy and reference.
  5. Compute Diffusion-DPO loss.
  6. Update LoRA parameters.

Supports both image and video modalities with accelerate for
mixed-precision and multi-GPU training.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm_dpo.config.experiment_config import ExperimentConfig
from vlm_dpo.training.dpo_loss import DiffusionDPOLoss
from vlm_dpo.training.scheduler import compute_log_prob, sample_timesteps, add_noise

logger = logging.getLogger(__name__)


class DPOTrainer:
    """
    Trainer for Diffusion-DPO with LoRA.

    Args:
        config: Full experiment configuration.
        policy_unet: The denoising model with LoRA adapters (trainable).
        ref_unet: Frozen copy of the denoising model (reference).
        vae: VAE for encoding images/videos to latent space.
        text_encoder: Text encoder for conditioning.
        tokenizer: Tokenizer for the text encoder.
        noise_scheduler: Diffusion noise scheduler.
        train_dataset: PreferenceDataset with winner/loser pairs.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        policy_unet: nn.Module,
        ref_unet: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        noise_scheduler: Any,
        train_dataset: Any,
    ):
        self.config = config
        self.policy_unet = policy_unet
        self.ref_unet = ref_unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.train_dataset = train_dataset

        # Freeze reference model
        self.ref_unet.eval()
        for p in self.ref_unet.parameters():
            p.requires_grad = False

        # Freeze VAE and text encoder
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # DPO loss
        self.dpo_loss = DiffusionDPOLoss(
            beta=config.dpo.beta,
            label_smoothing=config.dpo.label_smoothing,
            reward_weights=(
                config.scoring.reward_weights
                if config.scoring.strategy == "multi_aspect"
                else None
            ),
        )

        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        # Setup accelerate
        self.accelerator = self._setup_accelerator()

        # Output directory
        self.output_dir = Path(config.eval.eval_output_dir).parent / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.global_step = 0
        self.best_accuracy = 0.0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer for LoRA parameters only."""
        trainable_params = [p for p in self.policy_unet.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Build a cosine LR scheduler with warmup."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.training.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.max_steps - self.config.training.warmup_steps,
            eta_min=self.config.training.learning_rate * 0.01,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.training.warmup_steps],
        )

    def _setup_accelerator(self) -> Any:
        """Initialize HuggingFace Accelerate."""
        from accelerate import Accelerator

        accelerator = Accelerator(
            mixed_precision=self.config.training.mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            log_with="wandb" if self.config.training.wandb_project else None,
        )

        if self.config.training.wandb_project:
            accelerator.init_trackers(
                project_name=self.config.training.wandb_project,
                config={
                    "experiment": self.config.name,
                    "beta": self.config.dpo.beta,
                    "lora_rank": self.config.lora.rank,
                    "lr": self.config.training.learning_rate,
                },
                init_kwargs={"wandb": {"name": self.config.training.wandb_run_name}},
            )

        return accelerator

    def _encode_to_latents(self, media: torch.Tensor) -> torch.Tensor:
        """
        Encode images/videos to latent space using the VAE.

        Args:
            media: Tensor of images (B, C, H, W) or videos (B, T, C, H, W).

        Returns:
            Latent tensors.
        """
        with torch.no_grad():
            if media.ndim == 5:
                # Video: encode frame by frame and stack
                b, t, c, h, w = media.shape
                media_flat = media.reshape(b * t, c, h, w)
                latents_flat = self.vae.encode(media_flat).latent_dist.sample()
                latents_flat = latents_flat * self.vae.config.scaling_factor
                _, cl, hl, wl = latents_flat.shape
                latents = latents_flat.reshape(b, t, cl, hl, wl)
            else:
                # Image
                latents = self.vae.encode(media).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        return latents

    def _encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        """Encode text prompts to conditioning embeddings."""
        with torch.no_grad():
            inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
            encoder_output = self.text_encoder(**inputs)
            return encoder_output.last_hidden_state

    def _training_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Execute a single DPO training step.

        Args:
            batch: Dict with "winner", "loser" tensors, "prompt" strings.

        Returns:
            Dict of loss and metric values.
        """
        winner = batch["winner"].to(self.accelerator.device)
        loser = batch["loser"].to(self.accelerator.device)
        prompts = batch["prompt"]

        # Encode to latents
        winner_latents = self._encode_to_latents(winner)
        loser_latents = self._encode_to_latents(loser)

        # Encode prompts
        encoder_hidden_states = self._encode_prompt(prompts)

        # Sample random timesteps
        bsz = winner_latents.shape[0]
        timesteps = sample_timesteps(
            bsz,
            self.noise_scheduler.config.num_train_timesteps,
            self.accelerator.device,
        )

        # Add noise
        noise_w = torch.randn_like(winner_latents)
        noise_l = torch.randn_like(loser_latents)

        noisy_winner = add_noise(winner_latents, noise_w, timesteps, self.noise_scheduler)
        noisy_loser = add_noise(loser_latents, noise_l, timesteps, self.noise_scheduler)

        # Compute policy log-probs
        policy_logp_w = compute_log_prob(
            self.policy_unet, noisy_winner, noise_w, timesteps,
            encoder_hidden_states, self.noise_scheduler,
        )
        policy_logp_l = compute_log_prob(
            self.policy_unet, noisy_loser, noise_l, timesteps,
            encoder_hidden_states, self.noise_scheduler,
        )

        # Compute reference log-probs (no gradients)
        with torch.no_grad():
            ref_logp_w = compute_log_prob(
                self.ref_unet, noisy_winner, noise_w, timesteps,
                encoder_hidden_states, self.noise_scheduler,
            )
            ref_logp_l = compute_log_prob(
                self.ref_unet, noisy_loser, noise_l, timesteps,
                encoder_hidden_states, self.noise_scheduler,
            )

        # Compute DPO loss
        loss_dict = self.dpo_loss(
            policy_logps_winner=policy_logp_w,
            policy_logps_loser=policy_logp_l,
            ref_logps_winner=ref_logp_w,
            ref_logps_loser=ref_logp_l,
        )

        return {
            "loss": loss_dict["loss"],
            "accuracy": loss_dict["accuracy"].item(),
            "reward_margin": loss_dict["reward_margin"].mean().item(),
        }

    def train(self) -> dict[str, Any]:
        """
        Run the full DPO training loop.

        Returns:
            Dict with final training metrics.
        """
        logger.info(f"Starting DPO training: {self.config.name}")
        logger.info(f"  Max steps: {self.config.training.max_steps}")
        logger.info(f"  Beta: {self.config.dpo.beta}")
        logger.info(f"  LoRA rank: {self.config.lora.rank}")
        logger.info(f"  LR: {self.config.training.learning_rate}")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        # Prepare with accelerate
        self.policy_unet, self.optimizer, dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.policy_unet, self.optimizer, dataloader, self.lr_scheduler
            )
        )
        self.ref_unet = self.ref_unet.to(self.accelerator.device)

        # Training loop
        self.policy_unet.train()
        metrics_history = []
        start_time = time.time()

        progress = tqdm(
            total=self.config.training.max_steps,
            desc="DPO Training",
            disable=not self.accelerator.is_local_main_process,
        )

        while self.global_step < self.config.training.max_steps:
            for batch in dataloader:
                if self.global_step >= self.config.training.max_steps:
                    break

                with self.accelerator.accumulate(self.policy_unet):
                    step_metrics = self._training_step(batch)
                    loss = step_metrics["loss"]

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.policy_unet.parameters(),
                            self.config.training.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1
                progress.update(1)

                # Logging
                if self.global_step % self.config.training.log_steps == 0:
                    log_data = {
                        "train/loss": loss.item(),
                        "train/accuracy": step_metrics["accuracy"],
                        "train/reward_margin": step_metrics["reward_margin"],
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/step": self.global_step,
                    }
                    self.accelerator.log(log_data, step=self.global_step)
                    metrics_history.append(log_data)

                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{step_metrics['accuracy']:.2%}",
                    )

                # Checkpointing
                if (
                    self.global_step % self.config.training.save_steps == 0
                    and self.accelerator.is_main_process
                ):
                    self._save_checkpoint()

                    # Track best
                    if step_metrics["accuracy"] > self.best_accuracy:
                        self.best_accuracy = step_metrics["accuracy"]
                        self._save_checkpoint(tag="best")

        progress.close()

        # Final save
        if self.accelerator.is_main_process:
            self._save_checkpoint(tag="final")

        elapsed = time.time() - start_time
        logger.info(
            f"Training complete in {elapsed / 60:.1f} min. "
            f"Best accuracy: {self.best_accuracy:.2%}"
        )

        # Save metrics history
        if self.accelerator.is_main_process:
            metrics_path = self.output_dir / "metrics_history.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_history, f, indent=2)

        self.accelerator.end_training()

        return {
            "final_loss": loss.item(),
            "best_accuracy": self.best_accuracy,
            "total_steps": self.global_step,
            "elapsed_minutes": elapsed / 60,
        }

    def _save_checkpoint(self, tag: str | None = None) -> None:
        """Save a training checkpoint."""
        if tag:
            ckpt_dir = self.output_dir / f"checkpoint-{tag}"
        else:
            ckpt_dir = self.output_dir / f"checkpoint-{self.global_step}"

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        unwrapped_model = self.accelerator.unwrap_model(self.policy_unet)
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(ckpt_dir)
        else:
            torch.save(
                {k: v for k, v in unwrapped_model.state_dict().items() if "lora_" in k},
                ckpt_dir / "lora_weights.pt",
            )

        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.lr_scheduler.state_dict(),
                "best_accuracy": self.best_accuracy,
            },
            ckpt_dir / "training_state.pt",
        )

        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def load_checkpoint(self, ckpt_dir: str | Path) -> None:
        """Resume training from a checkpoint."""
        ckpt_dir = Path(ckpt_dir)

        # Load LoRA weights
        lora_path = ckpt_dir / "lora_weights.pt"
        if lora_path.exists():
            lora_state = torch.load(lora_path, map_location="cpu")
            self.policy_unet.load_state_dict(lora_state, strict=False)
        elif hasattr(self.policy_unet, "load_adapter"):
            self.policy_unet.load_adapter(str(ckpt_dir))

        # Load training state
        state_path = ckpt_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.global_step = state["global_step"]
            self.optimizer.load_state_dict(state["optimizer_state"])
            self.lr_scheduler.load_state_dict(state["scheduler_state"])
            self.best_accuracy = state["best_accuracy"]

        logger.info(f"Resumed from {ckpt_dir} (step {self.global_step})")
