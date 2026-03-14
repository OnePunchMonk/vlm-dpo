"""
VLM-DPO command-line interface.

Subcommands:
    vlm-dpo generate  — Generate and score preference pairs
    vlm-dpo train     — Run DPO training
    vlm-dpo evaluate  — Compute evaluation metrics
    vlm-dpo score     — Standalone VLM scoring
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vlm_dpo")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    """Generate preference pairs and score with VLM."""
    from vlm_dpo.config import load_config
    from vlm_dpo.models import load_wan21, load_flux2, load_internvl
    from vlm_dpo.scoring import VLMScorer
    from vlm_dpo.data import PromptDataset, PairGenerator

    config = load_config(args.config, overrides=args.overrides)
    logger.info(f"Generating preference data for: {config.name}")

    # Create example prompts if none exist
    prompt_path = Path(config.data.prompt_file)
    if not prompt_path.exists():
        logger.info("No prompt file found, creating example prompts...")
        PromptDataset.create_example_file(prompt_path)

    # Load prompts
    prompts_ds = PromptDataset(prompt_path, max_prompts=config.data.num_pairs)
    prompts = prompts_ds.get_prompt_texts()

    # Load models
    if config.model.modality == "image":
        pipeline = load_flux2(
            config.model.image_model_id,
            dtype=config.model.dtype,
        )
    else:
        pipeline = load_wan21(
            config.model.video_model_id,
            dtype=config.model.dtype,
        )

    vlm_model, vlm_tokenizer = load_internvl(
        config.model.vlm_model_id,
        dtype=config.model.dtype,
    )

    # Create scorer and generator
    scorer = VLMScorer(
        model=vlm_model,
        tokenizer=vlm_tokenizer,
        strategy=config.scoring.strategy,
        reward_weights=config.scoring.reward_weights,
        num_score_frames=config.scoring.num_score_frames,
    )

    generator = PairGenerator(
        pipeline=pipeline,
        scorer=scorer,
        output_dir=config.data.output_dir,
        modality=config.model.modality,
    )

    # Generate pairs
    height = config.data.image_height if config.model.modality == "image" else config.data.video_height
    width = config.data.image_width if config.model.modality == "image" else config.data.video_width

    generator.generate_pairs(
        prompts=prompts,
        num_frames=config.data.video_num_frames,
        height=height,
        width=width,
        num_inference_steps=config.data.num_inference_steps,
        guidance_scale=config.data.guidance_scale,
        scoring_strategy=config.scoring.strategy,
    )

    logger.info("Preference data generation complete!")


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Run DPO training."""
    import copy
    import torch
    from vlm_dpo.config import load_config
    from vlm_dpo.models import load_wan21, load_flux2, apply_lora
    from vlm_dpo.models.lora_utils import transfer_lora_weights
    from vlm_dpo.data import PreferenceDataset
    from vlm_dpo.training import DPOTrainer

    config = load_config(args.config, overrides=args.overrides)
    logger.info(f"Starting DPO training: {config.name}")

    # Load pipeline
    if config.model.modality == "image":
        pipeline = load_flux2(config.model.image_model_id, dtype=config.model.dtype)
    else:
        pipeline = load_wan21(config.model.video_model_id, dtype=config.model.dtype)

    # Extract components
    unet = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    noise_scheduler = pipeline.scheduler

    # Create reference (frozen copy)
    ref_unet = copy.deepcopy(unet)

    # Apply LoRA to policy
    policy_unet = apply_lora(
        unet,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )

    # Cross-modal transfer if configured
    if config.transfer.source_lora_path:
        logger.info(f"Loading source LoRA from: {config.transfer.source_lora_path}")
        src_state = torch.load(
            Path(config.transfer.source_lora_path) / "lora_weights.pt",
            map_location="cpu",
        )
        transfer_lora_weights(
            src_state,
            policy_unet,
            mapping_strategy=config.transfer.mapping_strategy,
        )

    # Load dataset
    dataset = PreferenceDataset(
        data_dir=config.data.output_dir,
        modality=config.model.modality,
        video_num_frames=config.data.video_num_frames,
    )

    # Create trainer and run
    trainer = DPOTrainer(
        config=config,
        policy_unet=policy_unet,
        ref_unet=ref_unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        train_dataset=dataset,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    results = trainer.train()
    logger.info(f"Training complete: {results}")


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation metrics."""
    import json
    from vlm_dpo.config import load_config
    from vlm_dpo.evaluation import compute_fid, compute_fvd, compute_clip_score, compute_cohens_kappa

    config = load_config(args.config, overrides=args.overrides)
    logger.info(f"Running evaluation for: {config.name}")

    results = {}
    output_dir = Path(config.eval.eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in config.eval.metrics:
        logger.info(f"Computing metric: {metric_name}")

        if metric_name == "fid":
            score = compute_fid(
                real_dir=args.real_dir or "data/real_images",
                gen_dir=args.gen_dir or output_dir / "generated",
            )
            results["fid"] = score

        elif metric_name == "fvd":
            logger.info("FVD requires pre-loaded video tensors — see evaluation API")
            results["fvd"] = "requires_manual_setup"

        elif metric_name == "clip_score":
            logger.info("CLIP score requires generated images + prompts — see evaluation API")
            results["clip_score"] = "requires_manual_setup"

        elif metric_name == "cohens_kappa":
            if config.eval.human_annotations_file:
                import numpy as np
                with open(config.eval.human_annotations_file) as f:
                    annotations = json.load(f)
                vlm = np.array(annotations.get("vlm_labels", []))
                human = np.array(annotations.get("human_labels", []))
                kappa_result = compute_cohens_kappa(vlm, human)
                results["cohens_kappa"] = kappa_result
            else:
                logger.warning("No human annotations file specified for Cohen's kappa")

        elif metric_name == "vbench":
            from vlm_dpo.evaluation.vbench_wrapper import VBenchEvaluator
            evaluator = VBenchEvaluator(
                dimensions=config.eval.vbench_dimensions,
            )
            vbench_results = evaluator.evaluate(
                video_dir=args.gen_dir or output_dir / "generated",
                output_dir=output_dir / "vbench",
            )
            results["vbench"] = vbench_results

    # Save results
    results_path = output_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Evaluation results saved to {results_path}")
    logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# Subcommand: score
# ---------------------------------------------------------------------------

def cmd_score(args: argparse.Namespace) -> None:
    """Score a single video/image with the VLM."""
    import torch
    from PIL import Image
    from vlm_dpo.config import load_config
    from vlm_dpo.models import load_internvl
    from vlm_dpo.scoring import VLMScorer

    config = load_config(args.config, overrides=args.overrides)

    vlm_model, vlm_tokenizer = load_internvl(
        config.model.vlm_model_id,
        dtype=config.model.dtype,
    )

    scorer = VLMScorer(
        model=vlm_model,
        tokenizer=vlm_tokenizer,
        strategy=args.strategy or config.scoring.strategy,
        reward_weights=config.scoring.reward_weights,
    )

    # Load media
    media_path = Path(args.media)
    if media_path.suffix in (".mp4", ".avi", ".mov", ".webm"):
        from decord import VideoReader, cpu
        import numpy as np

        vr = VideoReader(str(media_path), ctx=cpu(0))
        frames = vr.get_batch(range(len(vr))).asnumpy()
        media = torch.from_numpy(frames)
        modality = "video"
    else:
        media = Image.open(media_path)
        modality = "image"

    # Score
    strategy = args.strategy or config.scoring.strategy
    score_fn = {
        "holistic": scorer.score_holistic,
        "multi_aspect": scorer.score_multi_aspect,
        "cot": scorer.score_cot,
    }[strategy]

    result = score_fn(media, args.prompt, modality)

    import json
    print(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="vlm-dpo",
        description="VLM-DPO: Aligning Video Diffusion Models via VLM Preference Optimization",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate preference pairs")
    gen_parser.add_argument("--config", required=True, help="Path to YAML config")
    gen_parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides (dot notation)")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Run DPO training")
    train_parser.add_argument("--config", required=True, help="Path to YAML config")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    train_parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides")

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    eval_parser.add_argument("--config", required=True, help="Path to YAML config")
    eval_parser.add_argument("--real-dir", type=str, help="Directory of real samples")
    eval_parser.add_argument("--gen-dir", type=str, help="Directory of generated samples")
    eval_parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides")

    # --- score ---
    score_parser = subparsers.add_parser("score", help="Score a single media file")
    score_parser.add_argument("--config", required=True, help="Path to YAML config")
    score_parser.add_argument("--media", required=True, help="Path to video/image")
    score_parser.add_argument("--prompt", required=True, help="Generation prompt")
    score_parser.add_argument("--strategy", choices=["holistic", "multi_aspect", "cot"], help="Scoring strategy")
    score_parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides")

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    commands = {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "score": cmd_score,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        try:
            cmd_fn(args)
        except Exception as e:
            logger.error(f"Command '{args.command}' failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
