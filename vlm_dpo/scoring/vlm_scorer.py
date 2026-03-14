"""
VLM-based scorer using InternVL-U for automated preference labeling.

Supports three scoring strategies:
  - holistic: single overall quality score
  - multi_aspect: decomposed per-dimension scores
  - cot: chain-of-thought reasoning followed by scores
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np
import torch
from PIL import Image

from vlm_dpo.scoring.prompts import get_scoring_prompt, COMPARISON_PROMPT

logger = logging.getLogger(__name__)


class VLMScorer:
    """
    Automated video/image quality scorer using a Vision-Language Model.

    Uses InternVL-U to evaluate generated content across multiple quality
    dimensions, replacing expensive human annotation.

    Args:
        model: Loaded InternVL model.
        tokenizer: Corresponding tokenizer.
        strategy: Default scoring strategy ("holistic", "multi_aspect", "cot").
        reward_weights: Weights for multi-aspect score aggregation.
        num_score_frames: Number of video frames to sample for scoring.
        temperature: VLM generation temperature.
        max_tokens: Max tokens for VLM response.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        strategy: str = "multi_aspect",
        reward_weights: dict[str, float] | None = None,
        num_score_frames: int = 8,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.reward_weights = reward_weights or {
            "prompt_adherence": 0.3,
            "temporal_consistency": 0.3,
            "visual_quality": 0.2,
            "motion_naturalness": 0.2,
        }
        self.num_score_frames = num_score_frames
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(self, video: torch.Tensor | np.ndarray) -> list[Image.Image]:
        """
        Sample frames uniformly from a video tensor.

        Args:
            video: Video tensor of shape (T, H, W, C) or (T, C, H, W), values in [0, 255].

        Returns:
            List of PIL Images.
        """
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()

        t = video.shape[0]
        indices = np.linspace(0, t - 1, self.num_score_frames, dtype=int)

        frames = []
        for i in indices:
            frame = video[i]
            # Handle (C, H, W) format
            if frame.shape[0] in (1, 3) and frame.ndim == 3:
                frame = np.transpose(frame, (1, 2, 0))
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))

        return frames

    # ------------------------------------------------------------------
    # VLM inference
    # ------------------------------------------------------------------

    def _query_vlm(self, images: list[Image.Image], text_prompt: str) -> str:
        """
        Send images + text to the VLM and return the text response.

        Args:
            images: List of PIL Images (frames or a single image).
            text_prompt: The scoring prompt.

        Returns:
            Raw text response from the VLM.
        """
        # Build the multi-image prompt for InternVL
        # Convention: <image> tokens followed by the text
        image_tokens = "".join(["<image>\n"] * len(images))
        full_prompt = image_tokens + text_prompt

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
        )

        # Process images via the model's image processor
        if hasattr(self.model, "img_context_token_id"):
            # InternVL2 style
            pixel_values = self._process_images_internvl(images)
            inputs["pixel_values"] = pixel_values

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt from the response
        if full_prompt in response:
            response = response[len(full_prompt):].strip()

        return response

    def _process_images_internvl(self, images: list[Image.Image]) -> torch.Tensor:
        """Process images for InternVL-U input."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        tensors = [transform(img) for img in images]
        return torch.stack(tensors).to(
            device=self.model.device,
            dtype=next(self.model.parameters()).dtype,
        )

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """Extract JSON from VLM response, handling markdown code blocks."""
        # Try to extract JSON from code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try direct JSON parsing
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        raise ValueError(f"Could not parse JSON from VLM response: {response[:200]}")

    # ------------------------------------------------------------------
    # Public scoring methods
    # ------------------------------------------------------------------

    def score_holistic(
        self,
        media: torch.Tensor | np.ndarray | Image.Image,
        prompt: str,
        modality: str = "video",
    ) -> dict[str, Any]:
        """
        Score media with a single holistic quality score.

        Args:
            media: Video tensor or PIL Image.
            prompt: Generation prompt.
            modality: "video" or "image".

        Returns:
            Dict with "score" (float) and "brief_reason" (str).
        """
        if modality == "video":
            frames = self._extract_frames(media)
        else:
            frames = [media] if isinstance(media, Image.Image) else [Image.fromarray(media)]

        scoring_prompt = get_scoring_prompt("holistic", prompt, modality)
        response = self._query_vlm(frames, scoring_prompt)
        return self._parse_json_response(response)

    def score_multi_aspect(
        self,
        media: torch.Tensor | np.ndarray | Image.Image,
        prompt: str,
        modality: str = "video",
    ) -> dict[str, float]:
        """
        Score media across multiple quality dimensions.

        Args:
            media: Video tensor or PIL Image.
            prompt: Generation prompt.
            modality: "video" or "image".

        Returns:
            Dict with per-dimension scores (floats 1-10).
        """
        if modality == "video":
            frames = self._extract_frames(media)
        else:
            frames = [media] if isinstance(media, Image.Image) else [Image.fromarray(media)]

        scoring_prompt = get_scoring_prompt("multi_aspect", prompt, modality)
        response = self._query_vlm(frames, scoring_prompt)
        return self._parse_json_response(response)

    def score_cot(
        self,
        media: torch.Tensor | np.ndarray | Image.Image,
        prompt: str,
        modality: str = "video",
    ) -> dict[str, Any]:
        """
        Score media with chain-of-thought reasoning.

        Args:
            media: Video tensor or PIL Image.
            prompt: Generation prompt.
            modality: "video" or "image".

        Returns:
            Dict with "analysis" (str), per-dimension scores, and "overall" score.
        """
        if modality == "video":
            frames = self._extract_frames(media)
        else:
            frames = [media] if isinstance(media, Image.Image) else [Image.fromarray(media)]

        scoring_prompt = get_scoring_prompt("cot", prompt, modality)
        response = self._query_vlm(frames, scoring_prompt)
        return self._parse_json_response(response)

    def aggregate_score(self, aspect_scores: dict[str, float]) -> float:
        """
        Compute weighted aggregate score from multi-aspect scores.

        Args:
            aspect_scores: Dict with per-dimension scores.

        Returns:
            Weighted sum as a single float.
        """
        total = 0.0
        weight_sum = 0.0
        for dim, weight in self.reward_weights.items():
            if dim in aspect_scores:
                total += weight * aspect_scores[dim]
                weight_sum += weight

        if weight_sum > 0:
            return total / weight_sum  # Normalize
        return total

    def compare_pair(
        self,
        media_a: torch.Tensor | np.ndarray | Image.Image,
        media_b: torch.Tensor | np.ndarray | Image.Image,
        prompt: str,
        strategy: str | None = None,
        modality: str = "video",
    ) -> dict[str, Any]:
        """
        Compare two generated samples and determine winner/loser.

        Args:
            media_a: First sample.
            media_b: Second sample.
            prompt: Generation prompt.
            strategy: Override scoring strategy (uses self.strategy if None).
            modality: "video" or "image".

        Returns:
            Dict with "winner" index (0 or 1), per-sample scores, and metadata.
        """
        strategy = strategy or self.strategy

        # Score each sample independently
        score_fn = {
            "holistic": self.score_holistic,
            "multi_aspect": self.score_multi_aspect,
            "cot": self.score_cot,
        }[strategy]

        scores_a = score_fn(media_a, prompt, modality)
        scores_b = score_fn(media_b, prompt, modality)

        # Compute aggregate scores for comparison
        if strategy == "holistic":
            agg_a = scores_a.get("score", 0.0)
            agg_b = scores_b.get("score", 0.0)
        elif strategy in ("multi_aspect", "cot"):
            agg_a = self.aggregate_score(scores_a)
            agg_b = self.aggregate_score(scores_b)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        winner = 0 if agg_a >= agg_b else 1
        margin = abs(agg_a - agg_b)

        return {
            "winner": winner,
            "loser": 1 - winner,
            "score_a": agg_a,
            "score_b": agg_b,
            "margin": margin,
            "details_a": scores_a,
            "details_b": scores_b,
            "strategy": strategy,
        }
