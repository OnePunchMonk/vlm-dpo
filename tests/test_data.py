"""Tests for data pipeline components."""

import json
import pytest
from pathlib import Path


class TestPromptDataset:
    """Tests for the PromptDataset class."""

    @pytest.fixture
    def prompt_file(self, tmp_path):
        """Create a temporary JSONL prompts file."""
        prompts = [
            {"prompt": "A cat playing", "category": "animals", "id": "001"},
            {"prompt": "A sunset scene", "category": "scenes", "id": "002"},
            {"prompt": "A dog running", "category": "animals", "id": "003"},
            {"prompt": "A city at night", "category": "scenes", "id": "004"},
            {"prompt": "A flower blooming", "category": "nature", "id": "005"},
        ]
        path = tmp_path / "prompts.jsonl"
        with open(path, "w") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")
        return path

    def test_load_all_prompts(self, prompt_file):
        from vlm_dpo.data import PromptDataset

        ds = PromptDataset(prompt_file)
        assert len(ds) == 5

    def test_max_prompts(self, prompt_file):
        from vlm_dpo.data import PromptDataset

        ds = PromptDataset(prompt_file, max_prompts=3)
        assert len(ds) == 3

    def test_category_filter(self, prompt_file):
        from vlm_dpo.data import PromptDataset

        ds = PromptDataset(prompt_file, categories=["animals"])
        assert len(ds) == 2
        assert all(p["category"] == "animals" for p in ds.prompts)

    def test_get_prompt_texts(self, prompt_file):
        from vlm_dpo.data import PromptDataset

        ds = PromptDataset(prompt_file)
        texts = ds.get_prompt_texts()
        assert len(texts) == 5
        assert texts[0] == "A cat playing"

    def test_getitem(self, prompt_file):
        from vlm_dpo.data import PromptDataset

        ds = PromptDataset(prompt_file)
        item = ds[0]
        assert "prompt" in item
        assert item["prompt"] == "A cat playing"

    def test_shuffle(self, prompt_file):
        from vlm_dpo.data import PromptDataset

        ds_no_shuffle = PromptDataset(prompt_file, shuffle=False)
        ds_shuffle = PromptDataset(prompt_file, shuffle=True, seed=42)

        # Same items, possibly different order
        assert set(ds_no_shuffle.get_prompt_texts()) == set(ds_shuffle.get_prompt_texts())

    def test_missing_file_raises(self, tmp_path):
        from vlm_dpo.data import PromptDataset

        with pytest.raises(FileNotFoundError):
            PromptDataset(tmp_path / "nonexistent.jsonl")

    def test_iter_batches(self, prompt_file):
        from vlm_dpo.data import PromptDataset

        ds = PromptDataset(prompt_file)
        batches = list(ds.iter_batches(2))
        assert len(batches) == 3  # 5 items, batch_size=2 → 3 batches
        assert len(batches[0]) == 2
        assert len(batches[-1]) == 1

    def test_create_example_file(self, tmp_path):
        from vlm_dpo.data import PromptDataset

        path = tmp_path / "example_prompts.jsonl"
        PromptDataset.create_example_file(path)
        assert path.exists()

        ds = PromptDataset(path)
        assert len(ds) == 10


class TestPreferenceDataset:
    """Tests for the PreferenceDataset class."""

    @pytest.fixture
    def preference_dir(self, tmp_path):
        """Create a minimal preference dataset directory."""
        # Create pair directories with dummy images
        from PIL import Image
        import numpy as np

        for i in range(3):
            pair_dir = tmp_path / "pairs" / f"{i:04d}"
            pair_dir.mkdir(parents=True)

            # Create dummy images
            for name in ["winner.png", "loser.png"]:
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(pair_dir / name)

        # Create metadata
        metadata = []
        for i in range(3):
            metadata.append({
                "pair_id": f"{i:04d}",
                "prompt": f"Test prompt {i}",
                "winner_path": f"pairs/{i:04d}/winner.png",
                "loser_path": f"pairs/{i:04d}/loser.png",
                "scores": {"winner": 8.0, "loser": 5.0},
                "margin": 3.0,
                "strategy": "holistic",
            })

        with open(tmp_path / "metadata.jsonl", "w") as f:
            for m in metadata:
                f.write(json.dumps(m) + "\n")

        return tmp_path

    def test_load_preference_dataset(self, preference_dir):
        from vlm_dpo.data import PreferenceDataset

        ds = PreferenceDataset(preference_dir, modality="image")
        assert len(ds) == 3

    def test_getitem_returns_expected_keys(self, preference_dir):
        from vlm_dpo.data import PreferenceDataset

        ds = PreferenceDataset(preference_dir, modality="image", image_size=(32, 32))
        item = ds[0]

        assert "prompt" in item
        assert "winner" in item
        assert "loser" in item
        assert "margin" in item
        assert item["winner"].ndim == 3  # (C, H, W)
        assert item["loser"].ndim == 3

    def test_margin_filter(self, preference_dir):
        from vlm_dpo.data import PreferenceDataset

        # All pairs have margin=3.0, so filtering at 5.0 should exclude all
        ds = PreferenceDataset(preference_dir, modality="image", min_margin=5.0)
        assert len(ds) == 0

    def test_missing_metadata_raises(self, tmp_path):
        from vlm_dpo.data import PreferenceDataset

        with pytest.raises(FileNotFoundError):
            PreferenceDataset(tmp_path, modality="image")
