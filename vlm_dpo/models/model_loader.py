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
    from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, WanPipeline
    from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _get_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
) -> "WanPipeline":
    """
    Load the Wan2.1 text-to-video pipeline.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Torch dtype string.
        device_map: Device placement strategy.

    Returns:
        Loaded diffusers pipeline.
    """
    from diffusers import WanPipeline

    torch_dtype = _resolve_dtype(dtype)
    logger.info(f"Loading Wan2.1 from '{model_id}' (dtype={dtype})")

    pipeline = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    device = _get_device() if device_map == "auto" else device_map
    pipeline = pipeline.to(device)

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

    device = _get_device() if device_map == "auto" else device_map
    pipeline = pipeline.to(device)

    logger.info("CogVideoX pipeline loaded successfully")
    return pipeline


# ---------------------------------------------------------------------------
# Image model loader
# ---------------------------------------------------------------------------

def load_flux2(
    model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> "StableDiffusion3Pipeline":
    """
    Load the SD3-medium text-to-image pipeline for image DPO.

    Replaces Flux.1-dev (32B) with SD3-medium (~2B, ~8GB in bfloat16),
    which fits on Apple Silicon / 16GB GPUs and shares the DiT architecture
    with Wan2.1, making cross-modal LoRA transfer (Exp 4) more compatible.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Torch dtype string.
        device_map: Device placement strategy.

    Returns:
        Loaded SD3 pipeline.
    """
    from diffusers import StableDiffusion3Pipeline

    torch_dtype = _resolve_dtype(dtype)
    logger.info(f"Loading SD3-medium from '{model_id}' (dtype={dtype})")

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    device = _get_device() if device_map == "auto" else device_map
    pipeline = pipeline.to(device)

    logger.info("SD3-medium pipeline loaded successfully")
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

    # device_map="auto" uses accelerate dispatch (CUDA-only).
    # On MPS/CPU, load to CPU first then move to device.
    device = _get_device() if device_map == "auto" else device_map
    hf_device_map = "auto" if device == "cuda" else None

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=hf_device_map,
        trust_remote_code=True,
    )
    if hf_device_map is None:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model.eval()
    logger.info("InternVL model loaded successfully")
    return model, tokenizer
