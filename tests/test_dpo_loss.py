"""Tests for the Diffusion-DPO loss."""

import pytest
import torch

from vlm_dpo.training.dpo_loss import DiffusionDPOLoss


class TestDiffusionDPOLoss:
    """Unit tests for the DPO loss function."""

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def loss_fn(self):
        return DiffusionDPOLoss(beta=0.1, label_smoothing=0.0)

    @pytest.fixture
    def loss_fn_smoothed(self):
        return DiffusionDPOLoss(beta=0.1, label_smoothing=0.1)

    @pytest.fixture
    def multi_aspect_loss_fn(self):
        return DiffusionDPOLoss(
            beta=0.1,
            reward_weights={
                "prompt_adherence": 0.3,
                "temporal_consistency": 0.3,
                "visual_quality": 0.2,
                "motion_naturalness": 0.2,
            },
        )

    def _make_logps(self, batch_size, winner_better=True):
        """Create log-prob tensors where winner has higher policy advantage."""
        if winner_better:
            policy_logps_w = torch.randn(batch_size) + 1.0  # Higher
            policy_logps_l = torch.randn(batch_size) - 1.0  # Lower
        else:
            policy_logps_w = torch.randn(batch_size) - 1.0
            policy_logps_l = torch.randn(batch_size) + 1.0

        ref_logps_w = torch.randn(batch_size)
        ref_logps_l = torch.randn(batch_size)
        return policy_logps_w, policy_logps_l, ref_logps_w, ref_logps_l

    def test_loss_returns_correct_keys(self, loss_fn, batch_size):
        """Loss should return all expected metric keys."""
        args = self._make_logps(batch_size)
        result = loss_fn(*args)

        assert "loss" in result
        assert "reward_winner" in result
        assert "reward_loser" in result
        assert "reward_margin" in result
        assert "accuracy" in result

    def test_loss_is_scalar(self, loss_fn, batch_size):
        """Loss should be a scalar tensor."""
        args = self._make_logps(batch_size)
        result = loss_fn(*args)
        assert result["loss"].dim() == 0

    def test_loss_is_positive(self, loss_fn, batch_size):
        """DPO loss should always be positive (it's a negative log-sigmoid)."""
        args = self._make_logps(batch_size)
        result = loss_fn(*args)
        assert result["loss"].item() > 0

    def test_gradient_flows(self, loss_fn, batch_size):
        """Gradients should flow through the loss."""
        policy_w = torch.randn(batch_size, requires_grad=True)
        policy_l = torch.randn(batch_size, requires_grad=True)
        ref_w = torch.randn(batch_size)
        ref_l = torch.randn(batch_size)

        result = loss_fn(policy_w, policy_l, ref_w, ref_l)
        result["loss"].backward()

        assert policy_w.grad is not None
        assert policy_l.grad is not None
        assert not torch.all(policy_w.grad == 0)

    def test_accuracy_perfect_when_winner_better(self, loss_fn, batch_size):
        """When winner consistently has higher log-ratio, accuracy should be high."""
        policy_w = torch.ones(batch_size) * 5.0
        policy_l = torch.ones(batch_size) * -5.0
        ref_w = torch.zeros(batch_size)
        ref_l = torch.zeros(batch_size)

        result = loss_fn(policy_w, policy_l, ref_w, ref_l)
        assert result["accuracy"].item() == 1.0

    def test_label_smoothing_increases_loss(self, loss_fn, loss_fn_smoothed, batch_size):
        """Label smoothing should change the loss compared to hard labels."""
        args = self._make_logps(batch_size)
        loss_hard = loss_fn(*args)["loss"].item()
        loss_smooth = loss_fn_smoothed(*args)["loss"].item()

        # They should differ
        assert loss_hard != loss_smooth

    def test_multi_aspect_weighting(self, multi_aspect_loss_fn, batch_size):
        """Multi-aspect loss should accept and use aspect scores."""
        args = self._make_logps(batch_size)
        aspect_scores = {
            "prompt_adherence": torch.randn(batch_size),
            "temporal_consistency": torch.randn(batch_size),
            "visual_quality": torch.randn(batch_size),
            "motion_naturalness": torch.randn(batch_size),
        }

        result = multi_aspect_loss_fn(*args, aspect_scores=aspect_scores)
        assert result["loss"].item() > 0

    def test_beta_sensitivity(self, batch_size):
        """Higher beta should lead to different loss values."""
        args = self._make_logps(batch_size)

        loss_low_beta = DiffusionDPOLoss(beta=0.01)(*args)["loss"].item()
        loss_high_beta = DiffusionDPOLoss(beta=1.0)(*args)["loss"].item()

        assert loss_low_beta != loss_high_beta

    def test_numerical_stability(self, loss_fn):
        """Loss should not produce NaN or Inf for extreme values."""
        # Very large values
        result = loss_fn(
            torch.tensor([100.0, -100.0]),
            torch.tensor([-100.0, 100.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        )
        assert not torch.isnan(result["loss"])
        assert not torch.isinf(result["loss"])
