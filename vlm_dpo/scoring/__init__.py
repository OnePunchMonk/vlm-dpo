"""VLM-based preference scoring for video/image quality assessment."""

from vlm_dpo.scoring.vlm_scorer import VLMScorer
from vlm_dpo.scoring.prompts import (
    HOLISTIC_PROMPT,
    MULTI_ASPECT_PROMPT,
    COT_PROMPT,
    get_scoring_prompt,
)

__all__ = [
    "VLMScorer",
    "HOLISTIC_PROMPT",
    "MULTI_ASPECT_PROMPT",
    "COT_PROMPT",
    "get_scoring_prompt",
]
