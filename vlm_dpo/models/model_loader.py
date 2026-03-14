"""
Factory functions for loading diffusion and VLM models.

Each loader returns the pipeline/model and handles dtype, device placement,
and optional quantization for memory-constrained setups.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, FluxPipeline
    from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[dtype_str]


# ---------------------------------------------------------------------------
# Video model loaders
# ---------------------------------------------------------------------------

def load_wan21(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> "DiffusionPipeline":
    """
    Load the Wan2.1 text-to-video pipeline.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Torch dtype string.
        device_map: Device placement strategy.

    Returns:
        Loaded diffusers pipeline.
    """
    from diffusers import AutoPipelineForText2Video

    torch_dtype = _resolve_dtype(dtype)
    logger.info(f"Loading Wan2.1 from '{model_id}' (dtype={dtype})")

    pipeline = AutoPipelineForText2Video.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    if device_map == "auto":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device_map)

    logger.info("Wan2.1 pipeline loaded successfully")
    return pipeline


def load_cogvideox(
    model_id: str = "THUDM/CogVideoX-5b",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> "DiffusionPipeline":
    """
    Load the CogVideoX text-to-video pipeline.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Torch dtype string.
        device_map: Device placement strategy.

    Returns:
        Loaded diffusers pipeline.
    """
    from diffusers import CogVideoXPipeline

    torch_dtype = _resolve_dtype(dtype)
    logger.info(f"Loading CogVideoX from '{model_id}' (dtype={dtype})")

    pipeline = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    if device_map == "auto":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device_map)

    logger.info("CogVideoX pipeline loaded successfully")
    return pipeline


# ---------------------------------------------------------------------------
# Image model loader
# ---------------------------------------------------------------------------

def load_flux2(
    model_id: str = "black-forest-labs/FLUX.1-dev",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> "FluxPipeline":
    """
    Load the Flux.2 text-to-image pipeline.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Torch dtype string.
        device_map: Device placement strategy.

    Returns:
        Loaded Flux pipeline.
    """
    from diffusers import FluxPipeline

    torch_dtype = _resolve_dtype(dtype)
    logger.info(f"Loading Flux.2 from '{model_id}' (dtype={dtype})")

    pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    if device_map == "auto":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device_map)

    logger.info("Flux.2 pipeline loaded successfully")
    return pipeline


# ---------------------------------------------------------------------------
# VLM loader
# ---------------------------------------------------------------------------

def load_internvl(
    model_id: str = "OpenGVLab/InternVL2_5-4B",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple["AutoModel", "AutoTokenizer"]:
    """
    Load InternVL-U for preference scoring.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Torch dtype string.
        device_map: Device placement strategy.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModel, AutoTokenizer

    torch_dtype = _resolve_dtype(dtype)
    logger.info(f"Loading InternVL from '{model_id}' (dtype={dtype})")

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model.eval()
    logger.info("InternVL model loaded successfully")
    return model, tokenizer
