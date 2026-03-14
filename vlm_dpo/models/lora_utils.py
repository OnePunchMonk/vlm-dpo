"""
LoRA adapter utilities for applying, saving, and transferring adapters.

Handles cross-modal LoRA weight transfer (Image DiT → Video DiT).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
    bias: str = "none",
) -> nn.Module:
    """
    Apply LoRA adapters to a model via PEFT.

    Args:
        model: The base model to add LoRA adapters to.
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling factor.
        dropout: Dropout probability for LoRA layers.
        target_modules: List of module name patterns to target.
        bias: Bias handling strategy ("none", "all", "lora_only").

    Returns:
        Model wrapped with LoRA adapters.
    """
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["to_q", "to_v", "to_k", "to_out.0"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=None,  # Generic — works for diffusion DiTs
    )

    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return peft_model


@dataclass
class TransferReport:
    """Report from a cross-modal LoRA weight transfer."""
    transferred_layers: list[str]
    skipped_layers: list[str]
    shape_mismatches: list[str]
    total_transferred: int
    total_skipped: int

    def __str__(self) -> str:
        lines = [
            f"LoRA Transfer Report:",
            f"  Transferred: {self.total_transferred} layers",
            f"  Skipped:     {self.total_skipped} layers",
        ]
        if self.shape_mismatches:
            lines.append(f"  Shape mismatches: {len(self.shape_mismatches)}")
            for name in self.shape_mismatches[:5]:
                lines.append(f"    - {name}")
        return "\n".join(lines)


def transfer_lora_weights(
    src_state_dict: dict[str, torch.Tensor],
    dst_model: nn.Module,
    mapping_strategy: str = "name_match",
    layer_mapping: dict[str, str] | None = None,
) -> TransferReport:
    """
    Transfer LoRA weights from a source model to a destination model.

    This enables cross-modal transfer (e.g., image DiT → video DiT) by
    matching LoRA adapter weights between architectures.

    Args:
        src_state_dict: State dict from the source LoRA model.
        dst_model: Destination model (with LoRA applied).
        mapping_strategy: How to match layers — "name_match" or "position_match".
        layer_mapping: Optional explicit mapping {src_name: dst_name}.

    Returns:
        TransferReport with details of what was/wasn't transferred.
    """
    dst_state = dst_model.state_dict()

    # Filter to only LoRA parameters
    src_lora = {k: v for k, v in src_state_dict.items() if "lora_" in k}
    dst_lora_keys = {k for k in dst_state if "lora_" in k}

    transferred = []
    skipped = []
    mismatches = []

    if mapping_strategy == "name_match":
        # Direct name matching (possibly with layer_mapping renames)
        for src_key, src_val in src_lora.items():
            dst_key = layer_mapping.get(src_key, src_key) if layer_mapping else src_key

            if dst_key in dst_lora_keys:
                if dst_state[dst_key].shape == src_val.shape:
                    dst_state[dst_key] = src_val.clone()
                    transferred.append(dst_key)
                else:
                    mismatches.append(
                        f"{dst_key}: src={list(src_val.shape)} dst={list(dst_state[dst_key].shape)}"
                    )
                    skipped.append(dst_key)
            else:
                skipped.append(src_key)

    elif mapping_strategy == "position_match":
        # Match by position in the parameter list
        src_keys = sorted(src_lora.keys())
        dst_keys = sorted(dst_lora_keys)

        for i, (sk, dk) in enumerate(zip(src_keys, dst_keys)):
            src_val = src_lora[sk]
            if dst_state[dk].shape == src_val.shape:
                dst_state[dk] = src_val.clone()
                transferred.append(f"{sk} → {dk}")
            else:
                mismatches.append(
                    f"pos {i}: {sk}{list(src_val.shape)} → {dk}{list(dst_state[dk].shape)}"
                )
                skipped.append(dk)
    else:
        raise ValueError(f"Unknown mapping strategy: {mapping_strategy}")

    # Load the updated state dict
    dst_model.load_state_dict(dst_state, strict=False)

    report = TransferReport(
        transferred_layers=transferred,
        skipped_layers=skipped,
        shape_mismatches=mismatches,
        total_transferred=len(transferred),
        total_skipped=len(skipped),
    )
    logger.info(str(report))
    return report


def analyze_lora_transfer(
    src_state_dict: dict[str, torch.Tensor],
    dst_state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """
    Analyze compatibility between source and destination LoRA parameters.

    Useful for studying which DiT layers are modality-specific vs. shared.

    Args:
        src_state_dict: State dict from source model.
        dst_state_dict: State dict from destination model.

    Returns:
        Analysis dict with matched/unmatched layers and shape compatibility.
    """
    src_lora = {k: v.shape for k, v in src_state_dict.items() if "lora_" in k}
    dst_lora = {k: v.shape for k, v in dst_state_dict.items() if "lora_" in k}

    matched = []
    shape_compatible = []
    shape_incompatible = []
    src_only = []
    dst_only = []

    for key in src_lora:
        if key in dst_lora:
            matched.append(key)
            if src_lora[key] == dst_lora[key]:
                shape_compatible.append(key)
            else:
                shape_incompatible.append({
                    "key": key,
                    "src_shape": list(src_lora[key]),
                    "dst_shape": list(dst_lora[key]),
                })
        else:
            src_only.append(key)

    for key in dst_lora:
        if key not in src_lora:
            dst_only.append(key)

    return {
        "total_src_lora_params": len(src_lora),
        "total_dst_lora_params": len(dst_lora),
        "matched": len(matched),
        "shape_compatible": len(shape_compatible),
        "shape_incompatible": len(shape_incompatible),
        "src_only": len(src_only),
        "dst_only": len(dst_only),
        "compatible_layers": shape_compatible,
        "incompatible_layers": shape_incompatible,
        "src_only_layers": src_only,
        "dst_only_layers": dst_only,
    }
