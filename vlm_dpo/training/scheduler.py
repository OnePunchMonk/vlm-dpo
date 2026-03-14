"""
Noise scheduler utilities for Diffusion-DPO.

Computes log-probabilities under the diffusion forward process,
which are required for the DPO loss ratio terms:
  log π_θ(v|p)  and  log π_ref(v|p)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_log_prob(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    noise_scheduler: object,
) -> torch.Tensor:
    """
    Compute the log probability of a sample under the diffusion model.

    For diffusion models, this is computed as the negative MSE between the
    predicted noise and the actual noise at a given timestep, which serves
    as a proxy for the log-likelihood under the denoising model.

    In Diffusion-DPO (Wallace et al., 2023), we use:
        log π(x|c) ≈ -||ε_θ(x_t, t, c) - ε||²

    Args:
        model: The denoising UNet or DiT.
        noisy_latents: Noisy latent tensors x_t.
        noise: The actual noise ε added to get x_t.
        timesteps: Diffusion timesteps t.
        encoder_hidden_states: Conditioning embeddings.
        noise_scheduler: Diffusion noise scheduler (for SNR weighting).

    Returns:
        Log probability tensor of shape (batch_size,).
    """
    # Predict noise
    model_pred = model(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
    )

    # Handle different model output formats
    if hasattr(model_pred, "sample"):
        model_pred = model_pred.sample

    # Per-sample MSE (negative = log prob proxy)
    # Flatten spatial dims: (B, C, ...) → (B, -1)
    pred_flat = model_pred.reshape(model_pred.shape[0], -1)
    noise_flat = noise.reshape(noise.shape[0], -1)

    log_prob = -0.5 * (pred_flat - noise_flat).pow(2).mean(dim=-1)

    return log_prob


def sample_timesteps(
    batch_size: int,
    num_train_timesteps: int,
    device: torch.device,
    strategy: str = "uniform",
) -> torch.Tensor:
    """
    Sample timesteps for training.

    Args:
        batch_size: Number of timesteps to sample.
        num_train_timesteps: Total number of diffusion timesteps.
        device: Target device.
        strategy: Sampling strategy ("uniform", "importance", "low_snr").

    Returns:
        Tensor of timesteps, shape (batch_size,).
    """
    if strategy == "uniform":
        return torch.randint(0, num_train_timesteps, (batch_size,), device=device)

    elif strategy == "importance":
        # Bias towards mid-range timesteps where learning signal is strongest
        weights = torch.ones(num_train_timesteps, device=device)
        mid = num_train_timesteps // 2
        sigma = num_train_timesteps // 4
        t_range = torch.arange(num_train_timesteps, device=device, dtype=torch.float32)
        weights = torch.exp(-0.5 * ((t_range - mid) / sigma) ** 2)
        weights = weights / weights.sum()
        return torch.multinomial(weights, batch_size, replacement=True)

    elif strategy == "low_snr":
        # Bias towards high timesteps (low SNR) where fine details matter
        weights = torch.linspace(0.1, 1.0, num_train_timesteps, device=device)
        weights = weights / weights.sum()
        return torch.multinomial(weights, batch_size, replacement=True)

    else:
        raise ValueError(f"Unknown timestep sampling strategy: {strategy}")


def add_noise(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: object,
) -> torch.Tensor:
    """
    Add noise to latents according to the noise schedule.

    Args:
        latents: Clean latent tensors.
        noise: Random noise of same shape.
        timesteps: Timesteps at which to add noise.
        noise_scheduler: Diffusion noise scheduler.

    Returns:
        Noisy latents x_t.
    """
    return noise_scheduler.add_noise(latents, noise, timesteps)
