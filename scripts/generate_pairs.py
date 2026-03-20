#!/usr/bin/env python3
"""
Large-scale preference pair generation with checkpointing and resume.

Wraps the PairGenerator with:
  - Checkpointing: saves progress every N pairs so crashes don't lose work
  - Resume: detects already-generated pairs and skips them
  - Batched GPU management: loads/unloads models to fit in VRAM
  - Statistics: tracks score distributions, margins, timing

Usage:
    # Generate 5K video pairs from scratch
    python scripts/generate_pairs.py \
        --config configs/exp3_video_dpo.yaml \
        --output-dir data/exp3_video_pairs

    # Resume after a crash (picks up from last checkpoint)
    python scripts/generate_pairs.py \
        --config configs/exp3_video_dpo.yaml \
        --output-dir data/exp3_video_pairs \
        --resume

    # Generate only 500 pairs for Exp 1 validation
    python scripts/generate_pairs.py \
        --config configs/exp1_vlm_agreement.yaml \
        --output-dir data/exp1_vlm_agreement \
        --num-pairs 500
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_pairs")


def load_checkpoint(output_dir: Path) -> tuple[list[dict], set[str]]:
    """Load existing metadata and determine completed pair IDs."""
    metadata_path = output_dir / "metadata.jsonl"
    completed = []
    completed_ids = set()

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                completed.append(entry)
                completed_ids.add(entry["pair_id"])

    return completed, completed_ids


def append_metadata(output_dir: Path, entry: dict) -> None:
    """Append a single metadata entry to the JSONL file (incremental save)."""
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def flush_gpu():
    """Free GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_media(path: Path, modality: str) -> any:
    """Load saved media from disk for scoring.

    For videos, handles both .mp4 files and the frames-directory fallback
    (a directory of per-frame PNGs written when torchvision has no ffmpeg).
    """
    if modality == "image":
        from PIL import Image
        return Image.open(str(path)).convert("RGB")

    # Video: try mp4 first, then frames directory fallback
    frames_dir = path.with_suffix("")
    if frames_dir.is_dir():
        from PIL import Image
        import numpy as np
        frame_files = sorted(frames_dir.glob("*.png"))
        if not frame_files:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        frames = np.stack([np.array(Image.open(f)) for f in frame_files])
        return torch.from_numpy(frames)  # (T, H, W, C)

    try:
        from torchvision.io import read_video
        frames, _, _ = read_video(str(path), output_format="THWC", pts_unit="sec")
        return frames
    except Exception:
        import decord
        vr = decord.VideoReader(str(path))
        return torch.from_numpy(vr[:].asnumpy())


def main():
    parser = argparse.ArgumentParser(description="Generate preference pairs with checkpointing")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: from config)")
    parser.add_argument("--num-pairs", type=int, default=None,
                        help="Override number of pairs (default: from config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of pairs to generate before flushing GPU")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Log progress every N pairs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--scoring-strategy", type=str, default=None,
                        choices=["holistic", "multi_aspect", "cot"],
                        help="Override scoring strategy")
    args = parser.parse_args()

    # Load config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from vlm_dpo.config import load_config
    config = load_config(args.config)

    output_dir = Path(args.output_dir or config.data.output_dir)
    num_pairs = args.num_pairs or config.data.num_pairs
    scoring_strategy = args.scoring_strategy or config.scoring.strategy

    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target pairs: {num_pairs}")
    logger.info(f"Scoring strategy: {scoring_strategy}")

    # ---------------------------------------------------------------
    # Check for existing progress
    # ---------------------------------------------------------------
    completed_metadata, completed_ids = load_checkpoint(output_dir)
    start_idx = 0

    if args.resume and completed_ids:
        start_idx = len(completed_ids)
        logger.info(f"Resuming from pair {start_idx} ({len(completed_ids)} already done)")
    elif completed_ids and not args.resume:
        logger.warning(
            f"Found {len(completed_ids)} existing pairs in {output_dir}. "
            f"Use --resume to continue, or delete the directory to start fresh."
        )
        sys.exit(1)

    if start_idx >= num_pairs:
        logger.info(f"All {num_pairs} pairs already generated. Nothing to do.")
        sys.exit(0)

    # ---------------------------------------------------------------
    # Load prompts
    # ---------------------------------------------------------------
    from vlm_dpo.data import PromptDataset

    prompt_path = Path(config.data.prompt_file)
    if not prompt_path.exists():
        logger.info("Prompt file not found. Run `python scripts/prepare_prompts.py` first.")
        logger.info(f"Creating example prompts at {prompt_path} as fallback...")
        PromptDataset.create_example_file(prompt_path)

    prompts_ds = PromptDataset(prompt_path, max_prompts=num_pairs, shuffle=True, seed=args.seed)
    prompts = prompts_ds.get_prompt_texts()

    if len(prompts) < num_pairs:
        logger.warning(
            f"Only {len(prompts)} prompts available, but {num_pairs} pairs requested. "
            f"Prompts will be reused."
        )
        # Extend prompts by cycling
        import itertools
        prompts = list(itertools.islice(itertools.cycle(prompts), num_pairs))

    logger.info(f"Loaded {len(prompts)} prompts")

    modality = getattr(config.model, "modality", "video")
    ext = ".mp4" if modality == "video" else ".png"
    height = config.data.image_height if modality == "image" else config.data.video_height
    width = config.data.image_width if modality == "image" else config.data.video_width
    gen_kwargs = {
        "num_frames": config.data.video_num_frames,
        "height": height,
        "width": width,
        "num_inference_steps": config.data.num_inference_steps,
        "guidance_scale": config.data.guidance_scale,
    }

    stats = {"margins": [], "times": [], "errors": 0}
    t_start = time.time()

    # ---------------------------------------------------------------
    # Phase 1: Generate all raw samples with the diffusion pipeline
    # ---------------------------------------------------------------
    logger.info("=== Phase 1: Loading diffusion pipeline for generation ===")
    from vlm_dpo.models import load_wan21, load_flux2, load_internvl

    if modality == "image":
        pipeline = load_flux2(config.model.image_model_id, dtype=config.model.dtype)
    else:
        pipeline = load_wan21(config.model.video_model_id, dtype=config.model.dtype)

    from vlm_dpo.data import PairGenerator
    generator = PairGenerator(
        pipeline=pipeline,
        scorer=None,
        output_dir=str(output_dir),
        modality=modality,
    )

    # Track which pairs still need generation (may have partial progress on resume)
    pending_generation = []
    for idx in range(start_idx, num_pairs):
        pair_id = f"{idx:04d}"
        if pair_id in completed_ids:
            continue
        pair_dir = pairs_dir / pair_id
        sample_a_path = pair_dir / f"sample_a{ext}"
        sample_b_path = pair_dir / f"sample_b{ext}"
        if not (sample_a_path.exists() and sample_b_path.exists()):
            pending_generation.append(idx)

    for idx in tqdm(pending_generation, desc="Phase 1: Generating", total=len(pending_generation)):
        pair_id = f"{idx:04d}"
        prompt = prompts[idx]
        seed_a = args.seed + idx * 2
        seed_b = args.seed + idx * 2 + 1
        pair_dir = pairs_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        try:
            sample_a = generator._generate_sample(prompt, seed=seed_a, **gen_kwargs)
            sample_b = generator._generate_sample(prompt, seed=seed_b, **gen_kwargs)
            generator._save_media(sample_a, pair_dir / f"sample_a{ext}")
            generator._save_media(sample_b, pair_dir / f"sample_b{ext}")
        except Exception as e:
            logger.error(f"Pair {pair_id} generation failed: {e}")
            stats["errors"] += 1

        if (idx + 1) % (args.batch_size * 10) == 0:
            flush_gpu()

    logger.info("Phase 1 complete. Releasing diffusion pipeline from memory.")
    del pipeline, generator
    flush_gpu()

    # ---------------------------------------------------------------
    # Phase 2: Score all saved samples with the VLM
    # ---------------------------------------------------------------
    logger.info("=== Phase 2: Loading VLM scorer ===")
    vlm_model, vlm_tokenizer = load_internvl(
        config.model.vlm_model_id,
        dtype=config.model.dtype,
    )

    from vlm_dpo.scoring import VLMScorer
    scorer = VLMScorer(
        model=vlm_model,
        tokenizer=vlm_tokenizer,
        strategy=scoring_strategy,
        reward_weights=config.scoring.reward_weights,
        num_score_frames=config.scoring.num_score_frames,
    )

    pending_scoring = [
        idx for idx in range(start_idx, num_pairs)
        if f"{idx:04d}" not in completed_ids
    ]

    for idx in tqdm(pending_scoring, desc="Phase 2: Scoring", total=len(pending_scoring)):
        pair_id = f"{idx:04d}"
        prompt = prompts[idx]
        seed_a = args.seed + idx * 2
        seed_b = args.seed + idx * 2 + 1
        pair_dir = pairs_dir / pair_id

        sample_a_path = pair_dir / f"sample_a{ext}"
        sample_b_path = pair_dir / f"sample_b{ext}"

        if not (sample_a_path.exists() and sample_b_path.exists()):
            logger.warning(f"Pair {pair_id} missing generated samples, skipping scoring.")
            stats["errors"] += 1
            continue

        t_pair = time.time()
        try:
            sample_a = load_media(sample_a_path, modality)
            sample_b = load_media(sample_b_path, modality)

            comparison = scorer.compare_pair(
                sample_a, sample_b, prompt,
                strategy=scoring_strategy,
                modality=modality,
            )

            if comparison["winner"] == 0:
                winner_src, loser_src = sample_a_path, sample_b_path
                winner_seed, loser_seed = seed_a, seed_b
            else:
                winner_src, loser_src = sample_b_path, sample_a_path
                winner_seed, loser_seed = seed_b, seed_a

            winner_path = pair_dir / f"winner{ext}"
            loser_path = pair_dir / f"loser{ext}"

            # Rename both the file and its frames-dir fallback if present
            for src, dst in [(winner_src, winner_path), (loser_src, loser_path)]:
                src_frames = src.with_suffix("")
                dst_frames = dst.with_suffix("")
                if src.exists():
                    src.rename(dst)
                if src_frames.is_dir():
                    src_frames.rename(dst_frames)

            entry = {
                "pair_id": pair_id,
                "prompt": prompt,
                "winner_path": str(winner_path.relative_to(output_dir)),
                "loser_path": str(loser_path.relative_to(output_dir)),
                "scores": {
                    "winner": comparison[f"score_{'a' if comparison['winner'] == 0 else 'b'}"],
                    "loser": comparison[f"score_{'b' if comparison['winner'] == 0 else 'a'}"],
                },
                "margin": comparison["margin"],
                "strategy": comparison["strategy"],
                "seeds": {"winner_seed": winner_seed, "loser_seed": loser_seed},
            }

            append_metadata(output_dir, entry)
            completed_ids.add(pair_id)

            elapsed = time.time() - t_pair
            stats["margins"].append(comparison["margin"])
            stats["times"].append(elapsed)

        except Exception as e:
            logger.error(f"Pair {pair_id} scoring failed: {e}")
            stats["errors"] += 1
            continue

        if (idx + 1) % args.checkpoint_every == 0:
            done = len(completed_ids)
            avg_time = sum(stats["times"][-args.checkpoint_every:]) / max(len(stats["times"]), 1)
            remaining = (num_pairs - done) * avg_time
            logger.info(
                f"Progress: {done}/{num_pairs} pairs | "
                f"Avg time/pair: {avg_time:.1f}s | "
                f"Errors: {stats['errors']} | "
                f"ETA: {remaining/3600:.1f}h"
            )

    logger.info("Phase 2 complete. Releasing VLM from memory.")
    del vlm_model, vlm_tokenizer, scorer
    flush_gpu()

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    total_time = time.time() - t_start
    total_done = len(completed_ids)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Total pairs:    {total_done}")
    logger.info(f"Errors:         {stats['errors']}")
    logger.info(f"Total time:     {total_time/3600:.2f} hours")
    if stats["margins"]:
        logger.info(f"Avg margin:     {sum(stats['margins'])/len(stats['margins']):.3f}")
        logger.info(f"Min margin:     {min(stats['margins']):.3f}")
        logger.info(f"Max margin:     {max(stats['margins']):.3f}")
    if stats["times"]:
        logger.info(f"Avg time/pair:  {sum(stats['times'])/len(stats['times']):.1f}s")
    logger.info(f"Output:         {output_dir}")
    logger.info(f"Metadata:       {output_dir / 'metadata.jsonl'}")

    # Save stats summary
    stats_path = output_dir / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_pairs": total_done,
            "errors": stats["errors"],
            "total_time_hours": total_time / 3600,
            "avg_margin": sum(stats["margins"]) / len(stats["margins"]) if stats["margins"] else 0,
            "avg_time_per_pair": sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0,
        }, f, indent=2)


if __name__ == "__main__":
    main()
