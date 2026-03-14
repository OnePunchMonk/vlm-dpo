"""Tests for the config loading and merging system."""

from pathlib import Path
import pytest
import tempfile
import yaml


class TestExperimentConfig:
    """Test config loading, defaults resolution, and overrides."""

    @pytest.fixture
    def config_dir(self, tmp_path):
        """Create a temporary config directory with base + experiment configs."""
        # Base config
        base = {
            "model": {
                "video_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "dtype": "bfloat16",
            },
            "dpo": {"beta": 0.1, "label_smoothing": 0.0},
            "lora": {"rank": 16, "alpha": 32},
            "training": {"learning_rate": 1e-5, "max_steps": 2000},
            "data": {"num_pairs": 5000},
        }
        base_path = tmp_path / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base, f)

        # Experiment config that inherits from base
        exp = {
            "defaults": ["base"],
            "experiment": {"name": "test_exp", "description": "Test experiment"},
            "dpo": {"beta": 0.2},  # Override beta
            "data": {"num_pairs": 500},  # Override num_pairs
        }
        exp_path = tmp_path / "test_exp.yaml"
        with open(exp_path, "w") as f:
            yaml.dump(exp, f)

        return tmp_path

    def test_load_base_config(self, config_dir):
        """Loading the base config should use all defaults."""
        from vlm_dpo.config import load_config

        cfg = load_config(config_dir / "base.yaml")
        assert cfg.dpo.beta == 0.1
        assert cfg.lora.rank == 16
        assert cfg.data.num_pairs == 5000

    def test_load_with_defaults(self, config_dir):
        """Experiment config should inherit base and apply overrides."""
        from vlm_dpo.config import load_config

        cfg = load_config(config_dir / "test_exp.yaml")
        assert cfg.name == "test_exp"
        assert cfg.dpo.beta == 0.2  # Overridden
        assert cfg.data.num_pairs == 500  # Overridden
        assert cfg.lora.rank == 16  # Inherited from base

    def test_cli_overrides(self, config_dir):
        """CLI overrides should take priority over file values."""
        from vlm_dpo.config import load_config

        cfg = load_config(
            config_dir / "test_exp.yaml",
            overrides=["dpo.beta=0.5", "lora.rank=32"],
        )
        assert cfg.dpo.beta == 0.5
        assert cfg.lora.rank == 32

    def test_missing_config_raises(self):
        """Loading a non-existent config should raise FileNotFoundError."""
        from vlm_dpo.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_default_values(self):
        """ExperimentConfig defaults should be sane."""
        from vlm_dpo.config.experiment_config import ExperimentConfig

        cfg = ExperimentConfig()
        assert cfg.dpo.beta == 0.1
        assert cfg.lora.rank == 16
        assert cfg.scoring.strategy == "multi_aspect"
        assert len(cfg.scoring.reward_weights) == 4
        assert abs(sum(cfg.scoring.reward_weights.values()) - 1.0) < 1e-6
