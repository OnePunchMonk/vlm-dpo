"""
Prompt templates for VLM-based video/image quality scoring.

Three strategies:
  1. Holistic — single overall quality score
  2. Multi-aspect — decomposed scores per quality dimension
  3. CoT (Chain-of-Thought) — reasoning trace before scoring
"""

# ---------------------------------------------------------------------------
# Holistic scoring: single overall quality score (1-10)
# ---------------------------------------------------------------------------

HOLISTIC_PROMPT = """You are an expert video quality evaluator. Given a video generated from the prompt below, rate its overall quality on a scale of 1 to 10.

**Generation prompt:** "{prompt}"

Consider all aspects: how well it follows the prompt, visual quality, motion smoothness, temporal consistency, and aesthetics.

Respond with ONLY a JSON object:
{{"score": <float between 1.0 and 10.0>, "brief_reason": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Multi-aspect scoring: separate scores per dimension
# ---------------------------------------------------------------------------

MULTI_ASPECT_PROMPT = """You are an expert video quality evaluator. Given a video generated from the prompt below, rate it on each of the following dimensions (1-10 scale):

**Generation prompt:** "{prompt}"

**Dimensions:**
1. **Prompt Adherence** — Does the video accurately depict what the prompt describes?
2. **Temporal Consistency** — Are objects, characters, and scenes consistent across frames?
3. **Visual Quality** — Is the video sharp, well-lit, and free of artifacts?
4. **Motion Naturalness** — Are movements smooth, realistic, and physically plausible?

Respond with ONLY a JSON object:
{{
  "prompt_adherence": <float 1.0-10.0>,
  "temporal_consistency": <float 1.0-10.0>,
  "visual_quality": <float 1.0-10.0>,
  "motion_naturalness": <float 1.0-10.0>
}}"""


# ---------------------------------------------------------------------------
# Chain-of-Thought scoring: reasoning before final scores
# ---------------------------------------------------------------------------

COT_PROMPT = """You are an expert video quality evaluator. Given a video generated from the prompt below, analyze its quality step by step, then provide scores.

**Generation prompt:** "{prompt}"

**Instructions:**
1. First, describe what you see in the video (2-3 sentences).
2. Evaluate prompt adherence: does it match the prompt?
3. Evaluate temporal consistency: are there flickering, morphing, or disappearing elements?
4. Evaluate visual quality: sharpness, artifacts, lighting.
5. Evaluate motion naturalness: are movements smooth and physically plausible?
6. Provide final scores.

Respond with a JSON object:
{{
  "analysis": "<your step-by-step reasoning>",
  "prompt_adherence": <float 1.0-10.0>,
  "temporal_consistency": <float 1.0-10.0>,
  "visual_quality": <float 1.0-10.0>,
  "motion_naturalness": <float 1.0-10.0>,
  "overall": <float 1.0-10.0>
}}"""


# ---------------------------------------------------------------------------
# Image-specific variants (for Exp 2: Image DPO)
# ---------------------------------------------------------------------------

HOLISTIC_IMAGE_PROMPT = """You are an expert image quality evaluator. Given an image generated from the prompt below, rate its overall quality on a scale of 1 to 10.

**Generation prompt:** "{prompt}"

Consider: prompt adherence, visual quality, composition, aesthetics, and coherence.

Respond with ONLY a JSON object:
{{"score": <float between 1.0 and 10.0>, "brief_reason": "<one sentence>"}}"""


MULTI_ASPECT_IMAGE_PROMPT = """You are an expert image quality evaluator. Given an image generated from the prompt below, rate it on each dimension (1-10):

**Generation prompt:** "{prompt}"

**Dimensions:**
1. **Prompt Adherence** — Does the image accurately depict the prompt?
2. **Visual Quality** — Sharpness, detail, freedom from artifacts.
3. **Composition** — Layout, balance, focal point.
4. **Aesthetics** — Overall visual appeal and artistic quality.

Respond with ONLY a JSON object:
{{
  "prompt_adherence": <float 1.0-10.0>,
  "visual_quality": <float 1.0-10.0>,
  "composition": <float 1.0-10.0>,
  "aesthetics": <float 1.0-10.0>
}}"""


# ---------------------------------------------------------------------------
# Comparison prompt: which of two samples is better?
# ---------------------------------------------------------------------------

COMPARISON_PROMPT = """You are an expert {modality} quality evaluator. You are shown two {modality}s (A and B) generated from the same prompt. Determine which is better.

**Generation prompt:** "{prompt}"

Compare on: prompt adherence, {quality_dims}.

Respond with ONLY a JSON object:
{{
  "winner": "A" or "B",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence explanation>"
}}"""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "holistic": HOLISTIC_PROMPT,
    "multi_aspect": MULTI_ASPECT_PROMPT,
    "cot": COT_PROMPT,
}

_IMAGE_STRATEGY_MAP = {
    "holistic": HOLISTIC_IMAGE_PROMPT,
    "multi_aspect": MULTI_ASPECT_IMAGE_PROMPT,
    "cot": COT_PROMPT,  # CoT works for both
}


def get_scoring_prompt(
    strategy: str,
    prompt: str,
    modality: str = "video",
) -> str:
    """
    Get the formatted scoring prompt for a given strategy.

    Args:
        strategy: Scoring strategy ("holistic", "multi_aspect", "cot").
        prompt: The generation prompt to evaluate against.
        modality: "video" or "image".

    Returns:
        Formatted prompt string ready for VLM inference.
    """
    template_map = _IMAGE_STRATEGY_MAP if modality == "image" else _STRATEGY_MAP

    if strategy not in template_map:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(template_map)}")

    return template_map[strategy].format(prompt=prompt)
