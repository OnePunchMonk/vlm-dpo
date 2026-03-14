"""
Experiment configuration system for VLM-DPO.

Uses dataclasses for structured config and OmegaConf for YAML loading + CLI overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model identifiers and precision settings."""
    video_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    image_model_id: str = "black-forest-labs/FLUX.1-dev"
    vlm_model_id: str = "OpenGVLab/InternVL2_5-4B"
    secondary_video_model_id: str = "THUDM/CogVideoX-5b"
    # Which model to use as the primary target
    primary_model_id: str | None = None
    modality: str = "video"  # "video" or "image"
    dtype: str = "bfloat16"
    device_map: str = "auto"


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["to_q", "to_v", "to_k", "to_out.0"]
    )
    bias: str = "none"


@dataclass
class DPOConfig:
    """DPO training objective parameters."""
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_strategy: str = "frozen"  # "frozen" or "ema"


@dataclass
class ScoringConfig:
    """VLM scoring configuration."""
    strategy: str = "multi_aspect"  # "holistic", "multi_aspect", "cot"
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "prompt_adherence": 0.3,
            "temporal_consistency": 0.3,
            "visual_quality": 0.2,
            "motion_naturalness": 0.2,
        }
    )
    num_score_frames: int = 8
    vlm_temperature: float = 0.1
    vlm_max_tokens: int = 512
    ablation_strategies: list[str] | None = None


@dataclass
class TrainingConfig:
    """Training loop parameters."""
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    seed: int = 42
    save_steps: int = 500
    save_total_limit: int = 3
    log_steps: int = 10
    wandb_project: str = "vlm-dpo"
    wandb_run_name: str | None = None
    ablations: dict[str, Any] | None = None


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    prompt_file: str = "data/prompts.jsonl"
    num_pairs: int = 5000
    output_dir: str = "data/preference_pairs"
    video_num_frames: int = 16
    video_height: int = 480
    video_width: int = 848
    video_fps: int = 16
    image_height: int = 1024
    image_width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_workers: int = 4


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    metrics: list[str] = field(
        default_factory=lambda: ["fvd", "clip_score", "vbench"]
    )
    num_eval_samples: int = 100
    eval_output_dir: str = "outputs/eval"
    vbench_dimensions: list[str] = field(
        default_factory=lambda: [
            "subject_consistency",
            "motion_smoothness",
            "aesthetic_quality",
        ]
    )
    human_annotations_file: str | None = None
    baseline_results: str | None = None


@dataclass
class TransferConfig:
    """Cross-modal LoRA transfer configuration."""
    source_lora_path: str | None = None
    mapping_strategy: str = "name_match"  # "name_match", "position_match", "selective"
    transfer_layers: list[str] | None = None
    freeze_transferred_steps: int = 200


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "default"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _resolve_defaults(cfg_dict: DictConfig, config_dir: Path) -> DictConfig:
    """Resolve 'defaults' key by merging base configs."""
    if "defaults" not in cfg_dict:
        return cfg_dict

    merged = OmegaConf.create({})
    for default in cfg_dict.defaults:
        base_path = config_dir / f"{default}.yaml"
        if base_path.exists():
            base_cfg = OmegaConf.load(str(base_path))
            # Recursively resolve nested defaults
            base_cfg = _resolve_defaults(base_cfg, config_dir)
            merged = OmegaConf.merge(merged, base_cfg)

    # Remove the defaults key and merge overrides on top
    cfg_override = OmegaConf.create(
        {k: v for k, v in OmegaConf.to_container(cfg_dict).items() if k != "defaults"}
    )
    merged = OmegaConf.merge(merged, cfg_override)
    return merged


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """
    Load an experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional CLI overrides in dot-notation, e.g. ["dpo.beta=0.2"].

    Returns:
        Fully resolved ExperimentConfig dataclass.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_cfg = OmegaConf.load(str(config_path))
    config_dir = config_path.parent

    # Resolve defaults chain
    cfg = _resolve_defaults(raw_cfg, config_dir)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Handle the experiment name/description from the 'experiment' key
    experiment_name = "default"
    experiment_desc = ""
    if "experiment" in cfg:
        experiment_name = cfg.experiment.get("name", experiment_name)
        experiment_desc = cfg.experiment.get("description", experiment_desc)

    # Build structured config
    container = OmegaConf.to_container(cfg, resolve=True)

    def _build(cls, key):
        data = container.get(key, {})
        if data is None:
            data = {}
        # Filter to only valid fields
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})

    return ExperimentConfig(
        name=experiment_name,
        description=experiment_desc,
        model=_build(ModelConfig, "model"),
        lora=_build(LoRAConfig, "lora"),
        dpo=_build(DPOConfig, "dpo"),
        scoring=_build(ScoringConfig, "scoring"),
        training=_build(TrainingConfig, "training"),
        data=_build(DataConfig, "data"),
        eval=_build(EvalConfig, "eval"),
        transfer=_build(TransferConfig, "transfer"),
    )


def merge_configs(base: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """
    Merge override values into an existing ExperimentConfig.

    Args:
        base: The base config to update.
        overrides: Flat dict with dot-notation keys, e.g. {"dpo.beta": 0.2}.

    Returns:
        New ExperimentConfig with merged values.
    """
    base_dict = OmegaConf.structured(base)
    override_cfg = OmegaConf.create(overrides)
    merged = OmegaConf.merge(base_dict, override_cfg)
    return OmegaConf.to_object(merged)
