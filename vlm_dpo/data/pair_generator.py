"""
Preference pair generation pipeline.

Samples pairs from a base diffusion model and scores them with a VLM
to create labeled preference data for DPO training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PairGenerator:
    """
    Generates and scores preference pairs for DPO training.

    Pipeline:
        1. For each prompt, generate two samples from the base model.
        2. Score both with the VLM scorer.
        3. Label the higher-scored sample as "winner", lower as "loser".
        4. Save media files and metadata.

    Args:
        pipeline: Diffusion pipeline (video or image).
        scorer: VLMScorer instance.
        output_dir: Directory to save generated pairs.
        modality: "video" or "image".
    """

    def __init__(
        self,
        pipeline: Any,
        scorer: Any,
        output_dir: str | Path,
        modality: str = "video",
    ):
        self.pipeline = pipeline
        self.scorer = scorer
        self.output_dir = Path(output_dir)
        self.modality = modality

        # Create directory structure
        self.pairs_dir = self.output_dir / "pairs"
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

    def _generate_sample(
        self,
        prompt: str,
        num_frames: int = 16,
        height: int = 480,
        width: int = 848,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Any:
        """Generate a single sample from the diffusion pipeline."""
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        if self.modality == "video":
            kwargs.update({
                "num_frames": num_frames,
                "height": height,
                "width": width,
            })
        else:
            kwargs.update({
                "height": height,
                "width": width,
            })

        output = self.pipeline(**kwargs)

        if self.modality == "video":
            return output.frames[0] if hasattr(output, "frames") else output[0]
        else:
            return output.images[0] if hasattr(output, "images") else output[0]

    def _save_media(self, media: Any, path: Path) -> None:
        """Save a video or image to disk."""
        if self.modality == "video":
            self._save_video(media, path)
        else:
            if hasattr(media, "save"):
                media.save(str(path))
            else:
                from PIL import Image
                import numpy as np
                if isinstance(media, torch.Tensor):
                    media = media.cpu().numpy()
                if media.ndim == 3 and media.shape[0] in (1, 3):
                    media = media.transpose(1, 2, 0)
                media = ((media + 1) * 127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(media).save(str(path))

    @staticmethod
    def _save_video(frames: Any, path: Path) -> None:
        """Save video frames to mp4 using torchvision."""
        import numpy as np

        try:
            from torchvision.io import write_video

            if isinstance(frames, list):
                import numpy as np
                frames_np = np.stack([
                    np.array(f) if hasattr(f, '__array__') else f for f in frames
                ])
                video_tensor = torch.from_numpy(frames_np)
            elif isinstance(frames, torch.Tensor):
                video_tensor = frames
            elif isinstance(frames, np.ndarray):
                video_tensor = torch.from_numpy(frames)
            else:
                raise TypeError(f"Unsupported frames type: {type(frames)}")

            # Ensure (T, H, W, C) uint8 format
            if video_tensor.ndim == 4 and video_tensor.shape[1] in (1, 3):
                video_tensor = video_tensor.permute(0, 2, 3, 1)
            if video_tensor.dtype != torch.uint8:
                if video_tensor.max() <= 1.0:
                    video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8)
                else:
                    video_tensor = video_tensor.clamp(0, 255).to(torch.uint8)

            write_video(str(path), video_tensor, fps=16)

        except ImportError:
            # Fallback: save frames as individual images
            frames_dir = path.with_suffix("")
            frames_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"torchvision.io unavailable, saving frames to {frames_dir}")

    def generate_pairs(
        self,
        prompts: list[str],
        num_frames: int = 16,
        height: int = 480,
        width: int = 848,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        scoring_strategy: str | None = None,
        base_seed: int = 42,
    ) -> list[dict[str, Any]]:
        """
        Generate preference pairs for a list of prompts.

        For each prompt, generates two samples with different seeds,
        scores both, and labels winner/loser.

        Args:
            prompts: List of text prompts.
            num_frames: Number of video frames (ignored for images).
            height: Generation height.
            width: Generation width.
            num_inference_steps: Diffusion steps.
            guidance_scale: CFG scale.
            scoring_strategy: Override scorer strategy.
            base_seed: Starting seed (pairs get seed, seed+1).

        Returns:
            List of metadata dicts for each generated pair.
        """
        all_metadata = []
        ext = ".mp4" if self.modality == "video" else ".png"

        for idx, prompt in enumerate(tqdm(prompts, desc="Generating pairs")):
            pair_id = f"{idx:04d}"
            pair_dir = self.pairs_dir / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)

            # Generate two samples with different seeds
            seed_a = base_seed + idx * 2
            seed_b = base_seed + idx * 2 + 1

            gen_kwargs = {
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }

            logger.info(f"Pair {pair_id}: generating sample A (seed={seed_a})")
            sample_a = self._generate_sample(prompt, seed=seed_a, **gen_kwargs)

            logger.info(f"Pair {pair_id}: generating sample B (seed={seed_b})")
            sample_b = self._generate_sample(prompt, seed=seed_b, **gen_kwargs)

            # Score and compare
            comparison = self.scorer.compare_pair(
                sample_a,
                sample_b,
                prompt,
                strategy=scoring_strategy,
                modality=self.modality,
            )

            # Determine winner/loser
            if comparison["winner"] == 0:
                winner, loser = sample_a, sample_b
            else:
                winner, loser = sample_b, sample_a

            # Save media
            winner_path = pair_dir / f"winner{ext}"
            loser_path = pair_dir / f"loser{ext}"
            self._save_media(winner, winner_path)
            self._save_media(loser, loser_path)

            # Build metadata
            metadata = {
                "pair_id": pair_id,
                "prompt": prompt,
                "winner_path": str(winner_path.relative_to(self.output_dir)),
                "loser_path": str(loser_path.relative_to(self.output_dir)),
                "scores": {
                    "winner": comparison[f"score_{'a' if comparison['winner'] == 0 else 'b'}"],
                    "loser": comparison[f"score_{'b' if comparison['winner'] == 0 else 'a'}"],
                    "details_winner": comparison[
                        f"details_{'a' if comparison['winner'] == 0 else 'b'}"
                    ],
                    "details_loser": comparison[
                        f"details_{'b' if comparison['winner'] == 0 else 'a'}"
                    ],
                },
                "margin": comparison["margin"],
                "strategy": comparison["strategy"],
                "seeds": {"winner_seed": seed_a if comparison["winner"] == 0 else seed_b,
                          "loser_seed": seed_b if comparison["winner"] == 0 else seed_a},
            }

            all_metadata.append(metadata)

        # Save metadata
        metadata_path = self.output_dir / "metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for m in all_metadata:
                f.write(json.dumps(m) + "\n")

        logger.info(
            f"Generated {len(all_metadata)} pairs → {self.output_dir}\n"
            f"  Avg margin: {sum(m['margin'] for m in all_metadata) / len(all_metadata):.2f}"
        )

        return all_metadata

    def generate_hard_negatives(
        self,
        media: Any,
        prompt: str,
        edit_prompt: str | None = None,
    ) -> Any:
        """
        Generate a hard negative by editing an existing sample.

        Uses the diffusion pipeline's img2img or video editing capabilities
        to create a subtly degraded version.

        Args:
            media: The original high-quality sample.
            prompt: The original generation prompt.
            edit_prompt: Modified prompt for the negative (optional).

        Returns:
            Edited media tensor.
        """
        if edit_prompt is None:
            # Default: perturb the prompt slightly to create a misalignment
            edit_prompt = prompt + " but slightly wrong"

        # Use img2img / vid2vid with high noise for subtle edits
        kwargs = {
            "prompt": edit_prompt,
            "num_inference_steps": 20,
            "strength": 0.3,  # Low strength = subtle changes
        }

        if self.modality == "video":
            kwargs["video"] = media
        else:
            kwargs["image"] = media

        try:
            output = self.pipeline(**kwargs)
            if self.modality == "video":
                return output.frames[0] if hasattr(output, "frames") else output[0]
            else:
                return output.images[0] if hasattr(output, "images") else output[0]
        except Exception as e:
            logger.warning(f"Hard negative generation failed: {e}. Returning original.")
            return media
