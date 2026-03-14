"""Tests for the VLM scoring module."""

import json
import pytest
from unittest.mock import MagicMock, patch

import numpy as np
import torch


class TestVLMScorer:
    """Tests for the VLMScorer class."""

    @pytest.fixture
    def mock_vlm(self):
        """Create a mock VLM model and tokenizer."""
        model = MagicMock()
        model.device = torch.device("cpu")
        model.parameters.return_value = iter([torch.randn(1)])

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 20)),
            "attention_mask": torch.ones(1, 20),
        }

        return model, tokenizer

    @pytest.fixture
    def scorer(self, mock_vlm):
        from vlm_dpo.scoring import VLMScorer

        model, tokenizer = mock_vlm
        return VLMScorer(
            model=model,
            tokenizer=tokenizer,
            strategy="multi_aspect",
        )

    def test_extract_frames(self, scorer):
        """Frame extraction should return correct number of PIL images."""
        video = np.random.randint(0, 255, (32, 64, 64, 3), dtype=np.uint8)
        frames = scorer._extract_frames(video)
        assert len(frames) == scorer.num_score_frames

    def test_extract_frames_from_tensor(self, scorer):
        """Frame extraction should work with torch tensors."""
        video = torch.randint(0, 255, (32, 64, 64, 3))
        frames = scorer._extract_frames(video)
        assert len(frames) == scorer.num_score_frames

    def test_parse_json_response_direct(self, scorer):
        """Should parse a direct JSON string."""
        response = '{"score": 7.5, "brief_reason": "Good quality"}'
        result = scorer._parse_json_response(response)
        assert result["score"] == 7.5

    def test_parse_json_response_code_block(self, scorer):
        """Should extract JSON from markdown code blocks."""
        response = '```json\n{"score": 8.0, "brief_reason": "Great"}\n```'
        result = scorer._parse_json_response(response)
        assert result["score"] == 8.0

    def test_parse_json_response_with_text(self, scorer):
        """Should extract JSON embedded in surrounding text."""
        response = 'Here is my evaluation: {"score": 6.5, "brief_reason": "OK"} end.'
        result = scorer._parse_json_response(response)
        assert result["score"] == 6.5

    def test_parse_json_response_invalid(self, scorer):
        """Should raise ValueError for non-JSON response."""
        with pytest.raises(ValueError):
            scorer._parse_json_response("This has no JSON at all")

    def test_aggregate_score(self, scorer):
        """Weighted aggregation should work correctly."""
        scores = {
            "prompt_adherence": 8.0,
            "temporal_consistency": 6.0,
            "visual_quality": 7.0,
            "motion_naturalness": 5.0,
        }
        agg = scorer.aggregate_score(scores)
        # Expected: (0.3*8 + 0.3*6 + 0.2*7 + 0.2*5) / 1.0 = 6.6
        assert abs(agg - 6.6) < 0.01

    def test_aggregate_score_partial(self, scorer):
        """Aggregation should handle missing dimensions gracefully."""
        scores = {"prompt_adherence": 8.0, "visual_quality": 7.0}
        agg = scorer.aggregate_score(scores)
        # Only prompt_adherence (0.3) and visual_quality (0.2) present
        # = (0.3*8 + 0.2*7) / (0.3 + 0.2) = 3.8 / 0.5 = 7.6
        assert abs(agg - 7.6) < 0.01


class TestScoringPrompts:
    """Tests for prompt template formatting."""

    def test_get_scoring_prompt_holistic(self):
        from vlm_dpo.scoring.prompts import get_scoring_prompt

        prompt = get_scoring_prompt("holistic", "A cat sitting on a mat", "video")
        assert "A cat sitting on a mat" in prompt
        assert "overall quality" in prompt

    def test_get_scoring_prompt_multi_aspect(self):
        from vlm_dpo.scoring.prompts import get_scoring_prompt

        prompt = get_scoring_prompt("multi_aspect", "A sunset over the ocean")
        assert "Prompt Adherence" in prompt
        assert "Temporal Consistency" in prompt

    def test_get_scoring_prompt_cot(self):
        from vlm_dpo.scoring.prompts import get_scoring_prompt

        prompt = get_scoring_prompt("cot", "A bird flying")
        assert "step by step" in prompt

    def test_get_scoring_prompt_image(self):
        from vlm_dpo.scoring.prompts import get_scoring_prompt

        prompt = get_scoring_prompt("holistic", "A mountain landscape", "image")
        assert "image quality" in prompt

    def test_invalid_strategy_raises(self):
        from vlm_dpo.scoring.prompts import get_scoring_prompt

        with pytest.raises(ValueError):
            get_scoring_prompt("invalid_strategy", "test prompt")
