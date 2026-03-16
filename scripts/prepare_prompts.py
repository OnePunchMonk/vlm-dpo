#!/usr/bin/env python3
"""
Prepare the prompt dataset for VLM-DPO experiments.

Creates a diverse JSONL prompt corpus covering multiple categories,
complexity levels, and motion types — designed to stress-test video
generation and expose quality differences for preference learning.

Usage:
    python scripts/prepare_prompts.py --output data/prompts.jsonl --num-prompts 5000
    python scripts/prepare_prompts.py --output data/prompts.jsonl --num-prompts 500 --categories animals nature
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt templates — designed for diverse video generation
# ---------------------------------------------------------------------------

SUBJECTS = {
    "animals": [
        "a golden retriever", "a tabby cat", "a bald eagle", "a red panda",
        "a dolphin", "a hummingbird", "a wolf", "a sea turtle", "a butterfly",
        "a horse", "a rabbit", "a flamingo", "a jellyfish", "a parrot",
        "a fox", "a deer", "a penguin", "a chameleon", "a koala", "an owl",
    ],
    "people": [
        "a dancer", "a chef", "a street musician", "a painter",
        "a martial artist", "a surfer", "a skateboarder", "a gymnast",
        "a blacksmith", "a potter", "a violinist", "a rock climber",
        "a figure skater", "a calligrapher", "a glassblower",
    ],
    "nature": [
        "ocean waves", "a waterfall", "cherry blossom petals", "northern lights",
        "a thunderstorm", "a mountain stream", "autumn leaves", "a coral reef",
        "a sand dune", "a volcanic eruption", "a snowflake forming", "a geyser",
        "a field of sunflowers", "a forest canopy", "rain on a pond",
    ],
    "scenes": [
        "a bustling Tokyo street", "a Venetian canal", "a Moroccan marketplace",
        "a rainy London street", "a New York subway station", "a Parisian cafe",
        "a Bangkok floating market", "a medieval castle courtyard",
        "a desert oasis", "a mountain village at dawn", "a fishing harbor",
        "a night market", "a train station platform", "a rooftop garden",
    ],
    "scifi": [
        "a futuristic cityscape", "a spaceship interior", "a robot",
        "a holographic display", "a terraformed Mars landscape",
        "a cyberpunk alley", "an alien marketplace", "a space station",
        "a mech suit", "a floating island", "a time portal", "a neon grid",
    ],
    "abstract": [
        "colorful smoke", "ink dissolving in water", "fractal patterns",
        "liquid metal morphing", "light refracting through crystal",
        "paint splatters", "soap bubbles", "aurora-like ribbons of light",
        "geometric shapes transforming", "particle systems",
    ],
}

ACTIONS = [
    "running", "walking slowly", "dancing gracefully", "spinning",
    "jumping", "swimming", "flying", "climbing", "falling gently",
    "emerging from shadow", "moving through fog", "reflected in water",
    "captured in slow motion", "accelerating forward", "floating weightlessly",
]

SETTINGS = [
    "on a sandy beach at sunset", "in a dense forest", "on a snowy mountain peak",
    "in a traditional Japanese garden", "under a starry night sky",
    "in a sunlit meadow", "inside an ancient temple", "on a city rooftop at night",
    "in heavy rain", "during golden hour", "in a foggy landscape",
    "at the edge of a cliff", "beside a calm lake", "in a neon-lit alley",
    "in an underwater cave", "in a field of wildflowers",
]

STYLES = [
    "", "cinematic", "photorealistic", "anime style", "watercolor painting style",
    "in the style of Studio Ghibli", "dramatic lighting", "macro photography",
    "aerial shot", "time-lapse", "slow motion", "black and white",
    "vintage film look", "4K ultra-realistic", "dreamy ethereal",
]

MOTION_TYPES = [
    "camera slowly panning left", "camera tracking forward",
    "camera zooming in", "camera orbiting around the subject",
    "dolly zoom effect", "steady static shot", "handheld camera movement",
    "birds eye view descending", "low angle looking up",
]

# Complex prompt templates for higher diversity
TEMPLATES = [
    "{subject} {action} {setting}, {style}",
    "{subject} {setting}, {motion}, {style}",
    "{subject} {action}, {style}, {motion}",
    "A close-up of {subject} {setting}, {style}",
    "A wide shot of {subject} {action} {setting}",
    "{subject} and {subject2} {setting}, {style}",
    "A timelapse of {subject} {setting}",
    "{subject} transitioning from day to night {setting}",
    "A dramatic shot of {subject} {action}, {motion}",
    "{subject} reflected in water {setting}, {style}",
]


def generate_prompt(rng: random.Random) -> dict:
    """Generate a single diverse prompt with metadata."""
    category = rng.choice(list(SUBJECTS.keys()))
    subject = rng.choice(SUBJECTS[category])
    action = rng.choice(ACTIONS)
    setting = rng.choice(SETTINGS)
    style = rng.choice(STYLES)
    motion = rng.choice(MOTION_TYPES)

    template = rng.choice(TEMPLATES)

    # Pick a second subject for dual-subject templates
    cat2 = rng.choice(list(SUBJECTS.keys()))
    subject2 = rng.choice(SUBJECTS[cat2])

    prompt = template.format(
        subject=subject,
        subject2=subject2,
        action=action,
        setting=setting,
        style=style,
        motion=motion,
    )

    # Clean up empty style/commas
    prompt = prompt.replace(", ,", ",").replace(",,", ",").strip(", ")
    # Capitalize first letter
    prompt = prompt[0].upper() + prompt[1:]

    # Estimate complexity
    word_count = len(prompt.split())
    if word_count <= 8:
        complexity = "simple"
    elif word_count <= 15:
        complexity = "medium"
    else:
        complexity = "complex"

    return {
        "prompt": prompt,
        "category": category,
        "complexity": complexity,
        "has_motion": "motion" in template or action in template,
    }


def generate_prompts(
    num_prompts: int,
    seed: int = 42,
    categories: list[str] | None = None,
) -> list[dict]:
    """Generate a diverse set of prompts."""
    rng = random.Random(seed)
    prompts = []
    seen = set()

    # Generate more than needed and deduplicate
    attempts = 0
    max_attempts = num_prompts * 5

    while len(prompts) < num_prompts and attempts < max_attempts:
        attempts += 1
        entry = generate_prompt(rng)

        if categories and entry["category"] not in categories:
            continue

        # Deduplicate by prompt text
        if entry["prompt"] in seen:
            continue
        seen.add(entry["prompt"])

        entry["id"] = f"{len(prompts):05d}"
        prompts.append(entry)

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Prepare prompt dataset for VLM-DPO")
    parser.add_argument("--output", type=str, default="data/prompts.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--num-prompts", type=int, default=5000,
                        help="Number of prompts to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Filter to specific categories (animals, people, nature, scenes, scifi, abstract)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_prompts} prompts (seed={args.seed})...")
    prompts = generate_prompts(
        num_prompts=args.num_prompts,
        seed=args.seed,
        categories=args.categories,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    # Print stats
    categories = {}
    complexities = {}
    for p in prompts:
        categories[p["category"]] = categories.get(p["category"], 0) + 1
        complexities[p["complexity"]] = complexities.get(p["complexity"], 0) + 1

    print(f"\nSaved {len(prompts)} prompts to {output_path}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} ({100*count/len(prompts):.1f}%)")
    print(f"\nComplexity distribution:")
    for comp, count in sorted(complexities.items()):
        print(f"  {comp}: {count} ({100*count/len(prompts):.1f}%)")


if __name__ == "__main__":
    main()
