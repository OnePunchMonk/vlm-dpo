"""Tests for LoRA transfer utilities."""

import pytest
import torch
import torch.nn as nn

from vlm_dpo.models.lora_utils import (
    analyze_lora_transfer,
    TransferReport,
)


class TestLoRATransferAnalysis:
    """Tests for cross-modal LoRA weight transfer analysis."""

    def _make_lora_state_dict(self, layers, shape=(16, 64)):
        """Create a fake LoRA state dict."""
        state = {}
        for layer in layers:
            state[f"{layer}.lora_A.weight"] = torch.randn(*shape)
            state[f"{layer}.lora_B.weight"] = torch.randn(shape[1], shape[0])
        return state

    def test_analyze_perfect_match(self):
        """All layers should match when both dicts have same structure."""
        layers = ["block.0.attn.to_q", "block.0.attn.to_v", "block.1.attn.to_q"]
        src = self._make_lora_state_dict(layers)
        dst = self._make_lora_state_dict(layers)

        result = analyze_lora_transfer(src, dst)

        assert result["matched"] == len(layers) * 2  # A and B for each
        assert result["shape_compatible"] == len(layers) * 2
        assert result["shape_incompatible"] == 0
        assert result["src_only"] == 0
        assert result["dst_only"] == 0

    def test_analyze_partial_match(self):
        """Should handle partially overlapping layer sets."""
        src = self._make_lora_state_dict(["block.0.to_q", "block.1.to_q"])
        dst = self._make_lora_state_dict(["block.0.to_q", "block.2.to_q"])

        result = analyze_lora_transfer(src, dst)

        assert result["matched"] == 2  # block.0.to_q A and B
        assert result["src_only"] == 2  # block.1.to_q A and B
        assert result["dst_only"] == 2  # block.2.to_q A and B

    def test_analyze_shape_mismatch(self):
        """Should detect shape incompatibilities."""
        src = {"layer.lora_A.weight": torch.randn(16, 64)}
        dst = {"layer.lora_A.weight": torch.randn(32, 64)}  # Different shape

        result = analyze_lora_transfer(src, dst)

        assert result["matched"] == 1
        assert result["shape_compatible"] == 0
        assert result["shape_incompatible"] == 1

    def test_analyze_empty_dicts(self):
        """Should handle empty state dicts gracefully."""
        result = analyze_lora_transfer({}, {})
        assert result["total_src_lora_params"] == 0
        assert result["total_dst_lora_params"] == 0


class TestTransferReport:
    """Tests for the TransferReport dataclass."""

    def test_report_str(self):
        report = TransferReport(
            transferred_layers=["layer1", "layer2"],
            skipped_layers=["layer3"],
            shape_mismatches=["layer4"],
            total_transferred=2,
            total_skipped=1,
        )
        s = str(report)
        assert "Transferred: 2" in s
        assert "Skipped:     1" in s
