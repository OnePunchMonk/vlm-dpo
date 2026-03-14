"""
Prompt dataset for loading text prompts used to generate video/image pairs.

Supports JSONL format with optional metadata (category, difficulty, etc.).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    """
    Dataset of text prompts for video/image generation.

    Expected JSONL format (one JSON object per line):
        {"prompt": "A dog running on a beach", "category": "animals", "id": "001"}
        {"prompt": "A city skyline at sunset",  "category": "scenes",  "id": "002"}

    The only required field is "prompt". All other fields are optional metadata.

    Args:
        prompt_file: Path to the JSONL file.
        max_prompts: Maximum number of prompts to load (None = all).
        categories: Optional list of categories to filter by.
        shuffle: Whether to shuffle prompts after loading.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        prompt_file: str | Path,
        max_prompts: int | None = None,
        categories: list[str] | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.prompt_file = Path(prompt_file)
        self.prompts: list[dict[str, Any]] = []

        self._load(max_prompts, categories)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.prompts)

        logger.info(f"Loaded {len(self.prompts)} prompts from {self.prompt_file}")

    def _load(
        self,
        max_prompts: int | None,
        categories: list[str] | None,
    ) -> None:
        """Load and filter prompts from the JSONL file."""
        if not self.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")

        with open(self.prompt_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue

                if "prompt" not in entry:
                    logger.warning(f"Skipping line {line_num}: missing 'prompt' field")
                    continue

                # Category filter
                if categories and entry.get("category") not in categories:
                    continue

                self.prompts.append(entry)

                if max_prompts and len(self.prompts) >= max_prompts:
                    break

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.prompts[idx]

    def get_prompt_texts(self) -> list[str]:
        """Get all prompt strings."""
        return [p["prompt"] for p in self.prompts]

    def iter_batches(self, batch_size: int) -> Iterator[list[dict[str, Any]]]:
        """Iterate over prompts in batches."""
        for i in range(0, len(self.prompts), batch_size):
            yield self.prompts[i : i + batch_size]

    @staticmethod
    def create_example_file(output_path: str | Path) -> None:
        """Create an example prompts file with diverse generation prompts."""
        prompts = [
            {"prompt": "A golden retriever running on a sandy beach at sunset", "category": "animals", "id": "001"},
            {"prompt": "A bustling city street in Tokyo at night with neon signs", "category": "scenes", "id": "002"},
            {"prompt": "A butterfly emerging from a chrysalis in slow motion", "category": "nature", "id": "003"},
            {"prompt": "An astronaut floating in space with Earth in the background", "category": "scifi", "id": "004"},
            {"prompt": "A chef preparing sushi in a traditional Japanese kitchen", "category": "people", "id": "005"},
            {"prompt": "Waves crashing against rocky cliffs during a storm", "category": "nature", "id": "006"},
            {"prompt": "A dancer performing ballet in an empty theater", "category": "people", "id": "007"},
            {"prompt": "A time-lapse of flowers blooming in a garden", "category": "nature", "id": "008"},
            {"prompt": "A futuristic car driving through a neon-lit city", "category": "scifi", "id": "009"},
            {"prompt": "A cat playing with a ball of yarn on a wooden floor", "category": "animals", "id": "010"},
        ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")

        logger.info(f"Created example prompts file at {output_path}")
