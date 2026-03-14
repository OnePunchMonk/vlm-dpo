"""
Preference dataset for DPO training.

Loads pre-scored winner/loser pairs and returns tensors suitable
for the Diffusion-DPO loss.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """
    PyTorch dataset of scored preference pairs for DPO training.

    Directory structure (created by PairGenerator):
        output_dir/
          metadata.jsonl          # One JSON per pair
          pairs/
            0000/
              winner.mp4 (or .png)
              loser.mp4  (or .png)
            0001/
              ...

    Each metadata entry:
        {
          "pair_id": "0000",
          "prompt": "...",
          "winner_path": "pairs/0000/winner.mp4",
          "loser_path": "pairs/0000/loser.mp4",
          "scores": {...},
          "strategy": "multi_aspect",
          "margin": 2.3,
        }

    Args:
        data_dir: Root directory containing metadata.jsonl and pairs/.
        modality: "video" or "image".
        video_num_frames: Number of frames to load per video.
        image_size: Target (H, W) for images.
        min_margin: Minimum score margin to include a pair (filter easy pairs).
        transform: Optional transform to apply to loaded media.
    """

    def __init__(
        self,
        data_dir: str | Path,
        modality: str = "video",
        video_num_frames: int = 16,
        image_size: tuple[int, int] = (512, 512),
        min_margin: float = 0.0,
        transform: Any = None,
    ):
        self.data_dir = Path(data_dir)
        self.modality = modality
        self.video_num_frames = video_num_frames
        self.image_size = image_size
        self.transform = transform

        self.pairs: list[dict[str, Any]] = []
        self._load_metadata(min_margin)

        logger.info(
            f"PreferenceDataset: {len(self.pairs)} pairs "
            f"(modality={modality}, min_margin={min_margin})"
        )

    def _load_metadata(self, min_margin: float) -> None:
        """Load and filter pair metadata."""
        metadata_path = self.data_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                # Filter by minimum margin
                if entry.get("margin", 0.0) < min_margin:
                    continue

                # Verify files exist
                winner_path = self.data_dir / entry["winner_path"]
                loser_path = self.data_dir / entry["loser_path"]
                if not winner_path.exists() or not loser_path.exists():
                    logger.warning(f"Missing files for pair {entry['pair_id']}, skipping")
                    continue

                self.pairs.append(entry)

    def _load_video(self, path: Path) -> torch.Tensor:
        """
        Load a video file and return as tensor (T, C, H, W).

        Uses decord for efficient video loading.
        """
        from decord import VideoReader, cpu

        vr = VideoReader(str(path), ctx=cpu(0))
        total_frames = len(vr)

        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, self.video_num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

        # Convert to (T, C, H, W) float tensor, normalize to [-1, 1]
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        tensor = tensor / 127.5 - 1.0

        return tensor

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load an image and return as tensor (C, H, W)."""
        from PIL import Image
        from torchvision import transforms

        img = Image.open(path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        return transform(img)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load a preference pair.

        Returns:
            Dict with keys:
                - "prompt": str
                - "winner": Tensor (T, C, H, W) for video or (C, H, W) for image
                - "loser": Tensor (same shape)
                - "scores": dict with per-aspect scores
                - "margin": float
        """
        entry = self.pairs[idx]

        winner_path = self.data_dir / entry["winner_path"]
        loser_path = self.data_dir / entry["loser_path"]

        if self.modality == "video":
            winner = self._load_video(winner_path)
            loser = self._load_video(loser_path)
        else:
            winner = self._load_image(winner_path)
            loser = self._load_image(loser_path)

        if self.transform:
            winner = self.transform(winner)
            loser = self.transform(loser)

        return {
            "prompt": entry["prompt"],
            "winner": winner,
            "loser": loser,
            "scores": entry.get("scores", {}),
            "margin": entry.get("margin", 0.0),
        }
