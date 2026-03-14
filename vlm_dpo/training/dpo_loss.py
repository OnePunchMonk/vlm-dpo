"""
Diffusion-DPO loss implementation.

Implements the DPO objective adapted for diffusion models:
    L = -log σ(β * [log π_θ(v_w|p)/π_ref(v_w|p) - log π_θ(v_l|p)/π_ref(v_l|p)])

Supports both holistic and multi-aspect reward decomposition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionDPOLoss(nn.Module):
    """
    DPO loss for diffusion models.

    Computes the DPO objective by comparing log-probability ratios between
    the policy model and a frozen reference model for winner/loser pairs.

    Args:
        beta: DPO temperature parameter — controls how strongly we
              optimize towards the preference. Lower = stronger.
        label_smoothing: Smoothing factor for the preference labels.
              0 = hard labels, >0 mixes in the reverse preference.
        reward_weights: Optional per-aspect weights for multi-aspect DPO.
              Keys should match the scoring dimensions. If None, uses
              holistic (single-score) DPO.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        reward_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reward_weights = reward_weights

    def forward(
        self,
        policy_logps_winner: torch.Tensor,
        policy_logps_loser: torch.Tensor,
        ref_logps_winner: torch.Tensor,
        ref_logps_loser: torch.Tensor,
        aspect_scores: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the Diffusion-DPO loss.

        Args:
            policy_logps_winner: Log probs of winner under policy, shape (B,).
            policy_logps_loser: Log probs of loser under policy, shape (B,).
            ref_logps_winner: Log probs of winner under reference, shape (B,).
            ref_logps_loser: Log probs of loser under reference, shape (B,).
            aspect_scores: Optional per-aspect margin scores for weighted DPO.
                Each value is shape (B,). Used only in multi-aspect mode.

        Returns:
            Dict with:
                - "loss": Scalar DPO loss.
                - "reward_winner": Implicit reward for winners, shape (B,).
                - "reward_loser": Implicit reward for losers, shape (B,).
                - "reward_margin": Winner - loser reward, shape (B,).
                - "accuracy": Fraction where winner reward > loser reward.
        """
        # Log-probability ratios
        winner_logratios = policy_logps_winner - ref_logps_winner
        loser_logratios = policy_logps_loser - ref_logps_loser

        # Implicit rewards
        reward_winner = self.beta * winner_logratios
        reward_loser = self.beta * loser_logratios

        # DPO logits (margin)
        logits = winner_logratios - loser_logratios

        # Apply multi-aspect weighting if configured
        if self.reward_weights and aspect_scores:
            aspect_weight = self._compute_aspect_weight(aspect_scores)
            logits = logits * aspect_weight

        # DPO loss with optional label smoothing
        if self.label_smoothing > 0:
            # Smoothed loss: mix positive and negative direction
            loss_pos = -F.logsigmoid(self.beta * logits)
            loss_neg = -F.logsigmoid(-self.beta * logits)
            loss = (1 - self.label_smoothing) * loss_pos + self.label_smoothing * loss_neg
        else:
            loss = -F.logsigmoid(self.beta * logits)

        loss = loss.mean()

        # Metrics
        reward_margin = (reward_winner - reward_loser).detach()
        accuracy = (reward_margin > 0).float().mean()

        return {
            "loss": loss,
            "reward_winner": reward_winner.detach(),
            "reward_loser": reward_loser.detach(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
        }

    def _compute_aspect_weight(
        self,
        aspect_scores: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute a per-sample weighting based on multi-aspect score margins.

        The idea: pairs where the margin is larger on important dimensions
        should contribute more to the loss.

        Args:
            aspect_scores: Dict mapping dimension name → margin tensor (B,).

        Returns:
            Weight tensor of shape (B,).
        """
        weight = torch.zeros_like(next(iter(aspect_scores.values())))

        total_w = 0.0
        for dim_name, dim_weight in self.reward_weights.items():
            if dim_name in aspect_scores:
                # Normalize margin to [0, 1] range
                margin = aspect_scores[dim_name]
                normalized = torch.sigmoid(margin)
                weight += dim_weight * normalized
                total_w += dim_weight

        if total_w > 0:
            weight = weight / total_w

        # Ensure minimum weight to avoid zero gradients
        weight = weight.clamp(min=0.1)

        return weight
