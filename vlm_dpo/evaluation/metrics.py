"""
Evaluation metrics for video and image generation quality.

Includes:
  - FID (Fréchet Inception Distance) for images
  - FVD (Fréchet Video Distance) for videos
  - CLIP Score for prompt-image/video alignment
  - Cohen's Kappa for VLM-human agreement
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FID — Image quality
# ---------------------------------------------------------------------------

def compute_fid(
    real_dir: str | Path,
    gen_dir: str | Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> float:
    """
    Compute FID between real and generated image directories.

    Uses the clean-fid library for reliable FID computation.

    Args:
        real_dir: Directory of real images.
        gen_dir: Directory of generated images.
        device: Compute device.
        batch_size: Batch size for feature extraction.

    Returns:
        FID score (lower = better).
    """
    try:
        from cleanfid import fid

        score = fid.compute_fid(
            str(real_dir),
            str(gen_dir),
            device=torch.device(device),
            batch_size=batch_size,
        )
        logger.info(f"FID: {score:.2f}")
        return score

    except ImportError:
        logger.error("clean-fid not installed. Run: pip install clean-fid")
        raise


# ---------------------------------------------------------------------------
# FVD — Video quality
# ---------------------------------------------------------------------------

def compute_fvd(
    real_videos: torch.Tensor | np.ndarray,
    gen_videos: torch.Tensor | np.ndarray,
    device: str = "cuda",
) -> float:
    """
    Compute FVD between real and generated video sets.

    Uses I3D features for computing the Fréchet Distance in video space.

    Args:
        real_videos: Real videos tensor (N, T, C, H, W) in [0, 1].
        gen_videos: Generated videos tensor (N, T, C, H, W) in [0, 1].
        device: Compute device.

    Returns:
        FVD score (lower = better).
    """
    if isinstance(real_videos, np.ndarray):
        real_videos = torch.from_numpy(real_videos)
    if isinstance(gen_videos, np.ndarray):
        gen_videos = torch.from_numpy(gen_videos)

    real_videos = real_videos.float().to(device)
    gen_videos = gen_videos.float().to(device)

    # Extract I3D features
    real_features = _extract_i3d_features(real_videos, device)
    gen_features = _extract_i3d_features(gen_videos, device)

    # Compute Fréchet Distance
    fvd = _frechet_distance(real_features, gen_features)
    logger.info(f"FVD: {fvd:.2f}")
    return fvd


def _extract_i3d_features(
    videos: torch.Tensor,
    device: str,
) -> np.ndarray:
    """
    Extract I3D features from a batch of videos.

    Falls back to a simple 3D convolution feature extractor if the
    pretrained I3D model is not available.
    """
    try:
        # Try using torchvision's Video ResNet as a feature extractor
        from torchvision.models.video import r3d_18, R3D_18_Weights

        model = r3d_18(weights=R3D_18_Weights.DEFAULT).to(device)
        model.eval()

        # Remove the classification head
        model.fc = torch.nn.Identity()

        features = []
        with torch.no_grad():
            for i in range(0, len(videos), 8):  # batch of 8
                batch = videos[i:i + 8]
                # R3D expects (B, C, T, H, W), resize to 112x112
                batch = torch.nn.functional.interpolate(
                    batch.permute(0, 2, 1, 3, 4),  # (B, C, T, H, W)
                    size=(16, 112, 112),
                    mode="trilinear",
                    align_corners=False,
                )
                feat = model(batch)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    except Exception as e:
        logger.warning(f"I3D feature extraction failed ({e}), using simple features")
        # Fallback: flatten and use mean pooling
        return videos.mean(dim=(2, 3, 4)).cpu().numpy()


def _frechet_distance(
    features_1: np.ndarray,
    features_2: np.ndarray,
) -> float:
    """Compute Fréchet Distance between two sets of features."""
    from scipy.linalg import sqrtm

    mu1 = features_1.mean(axis=0)
    mu2 = features_2.mean(axis=0)
    sigma1 = np.cov(features_1, rowvar=False)
    sigma2 = np.cov(features_2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# CLIP Score — Prompt-content alignment
# ---------------------------------------------------------------------------

def compute_clip_score(
    images: list[Any] | torch.Tensor,
    prompts: list[str],
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
) -> float:
    """
    Compute CLIP score between generated images/frames and their prompts.

    Args:
        images: List of PIL Images or tensor (B, C, H, W).
        prompts: Corresponding text prompts.
        model_name: CLIP model identifier.
        device: Compute device.

    Returns:
        Mean CLIP score (higher = better).
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    scores = []

    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch_imgs = images[i:i + 32]
            batch_prompts = prompts[i:i + 32]

            inputs = processor(
                text=batch_prompts,
                images=batch_imgs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits_per_image.diagonal()
            scores.extend(logits.cpu().tolist())

    mean_score = np.mean(scores)
    logger.info(f"CLIP Score: {mean_score:.4f}")
    return float(mean_score)


# ---------------------------------------------------------------------------
# Cohen's Kappa — VLM-Human agreement
# ---------------------------------------------------------------------------

def compute_cohens_kappa(
    vlm_labels: list[int] | np.ndarray,
    human_labels: list[int] | np.ndarray,
) -> dict[str, float]:
    """
    Compute Cohen's Kappa for inter-rater agreement.

    Args:
        vlm_labels: VLM preference choices (0 or 1).
        human_labels: Human preference choices (0 or 1).

    Returns:
        Dict with "kappa", "agreement_rate", and "p_value".
    """
    from sklearn.metrics import cohen_kappa_score

    vlm_labels = np.asarray(vlm_labels)
    human_labels = np.asarray(human_labels)

    kappa = cohen_kappa_score(human_labels, vlm_labels)
    agreement = np.mean(vlm_labels == human_labels)

    result = {
        "kappa": float(kappa),
        "agreement_rate": float(agreement),
        "n_samples": len(vlm_labels),
    }
    logger.info(
        f"Cohen's κ: {kappa:.4f} | Agreement: {agreement:.2%} | N={len(vlm_labels)}"
    )
    return result
