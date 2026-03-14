"""
VBench wrapper for standardized video quality evaluation.

Wraps the VBench library for consistent usage within the VLM-DPO pipeline.
"""

from __future__ import annotations

import logging
import subprocess
import json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VBenchEvaluator:
    """
    Wrapper around VBench for video quality evaluation.

    VBench provides fine-grained video quality metrics across multiple
    dimensions (subject consistency, motion smoothness, aesthetic quality, etc.).

    Args:
        dimensions: List of VBench dimensions to evaluate.
        device: Compute device.
    """

    # Available VBench dimensions
    ALL_DIMENSIONS = [
        "subject_consistency",
        "background_consistency",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        "object_class",
        "multiple_objects",
        "human_action",
        "color",
        "spatial_relationship",
        "scene",
        "temporal_style",
        "appearance_style",
        "overall_consistency",
    ]

    def __init__(
        self,
        dimensions: list[str] | None = None,
        device: str = "cuda",
    ):
        self.dimensions = dimensions or [
            "subject_consistency",
            "motion_smoothness",
            "aesthetic_quality",
        ]
        self.device = device

        # Validate dimensions
        for dim in self.dimensions:
            if dim not in self.ALL_DIMENSIONS:
                logger.warning(f"Unknown VBench dimension: {dim}")

    def evaluate(
        self,
        video_dir: str | Path,
        prompts: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, float]:
        """
        Evaluate videos in a directory using VBench.

        Args:
            video_dir: Directory containing generated videos.
            prompts: Optional list of prompts (for prompt-aware metrics).
            output_dir: Where to save VBench results.

        Returns:
            Dict mapping dimension names to scores.
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir) if output_dir else video_dir / "vbench_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._evaluate_python(video_dir, prompts, output_dir)
        except ImportError:
            logger.warning("VBench Python API not available, falling back to CLI")
            return self._evaluate_cli(video_dir, output_dir)

    def _evaluate_python(
        self,
        video_dir: Path,
        prompts: list[str] | None,
        output_dir: Path,
    ) -> dict[str, float]:
        """Evaluate using VBench Python API."""
        from vbench import VBench

        vbench = VBench(device=self.device, full_json_dir=str(output_dir))

        results = {}
        for dim in self.dimensions:
            logger.info(f"Evaluating VBench dimension: {dim}")
            try:
                score = vbench.evaluate(
                    videos_path=str(video_dir),
                    name=dim,
                    dimension_list=[dim],
                )
                results[dim] = float(score[dim]) if isinstance(score, dict) else float(score)
            except Exception as e:
                logger.error(f"VBench evaluation failed for {dim}: {e}")
                results[dim] = -1.0

        # Save results
        results_path = output_dir / "vbench_scores.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"VBench results saved to {results_path}")
        return results

    def _evaluate_cli(
        self,
        video_dir: Path,
        output_dir: Path,
    ) -> dict[str, float]:
        """Evaluate using VBench CLI (fallback)."""
        dims_str = ",".join(self.dimensions)

        cmd = [
            "python", "-m", "vbench.evaluate",
            "--videos_path", str(video_dir),
            "--dimension_list", dims_str,
            "--output_path", str(output_dir),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"VBench CLI output: {result.stdout}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"VBench CLI failed: {e}")
            return {dim: -1.0 for dim in self.dimensions}

        # Parse results
        results_path = output_dir / "vbench_scores.json"
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)

        return {dim: -1.0 for dim in self.dimensions}
