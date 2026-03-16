#!/usr/bin/env python3
"""
Prepare human evaluation for Experiment 1 (VLM agreement validation).

Takes generated video pairs + VLM scores and produces:
  1. HTML comparison pages for human annotation (A/B/Tie)
  2. Randomized presentation order (to avoid position bias)
  3. Annotation collection template (JSONL)
  4. Agreement computation after annotations are collected

Usage:
    # Step 1: Generate HTML for annotation
    python scripts/prepare_human_eval.py generate \
        --pairs-dir data/exp1_vlm_agreement \
        --output-dir outputs/exp1/human_eval \
        --num-pairs 500

    # Step 2: After collecting annotations, compute agreement
    python scripts/prepare_human_eval.py compute \
        --vlm-scores data/exp1_vlm_agreement/metadata.jsonl \
        --human-annotations outputs/exp1/human_eval/annotations.jsonl \
        --output outputs/exp1/agreement_results.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def generate_html_pages(
    pairs_dir: Path,
    output_dir: Path,
    num_pairs: int | None = None,
    seed: int = 42,
) -> None:
    """Generate HTML comparison pages for human annotation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = pairs_dir / "metadata.jsonl"
    if not metadata_path.exists():
        print(f"ERROR: metadata.jsonl not found in {pairs_dir}")
        print("Run pair generation first.")
        return

    # Load pair metadata
    pairs = []
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if num_pairs:
        pairs = pairs[:num_pairs]

    rng = random.Random(seed)

    # Randomize presentation order (A/B swap) to avoid position bias
    annotations_template = []
    for pair in pairs:
        swap = rng.random() < 0.5
        annotations_template.append({
            "pair_id": pair["pair_id"],
            "prompt": pair["prompt"],
            "video_left": pair["loser_path"] if swap else pair["winner_path"],
            "video_right": pair["winner_path"] if swap else pair["loser_path"],
            "swapped": swap,
            "vlm_winner": "right" if swap else "left",
            "vlm_margin": pair["margin"],
            "human_label": None,  # To be filled by annotator
        })

    # Save annotation template
    template_path = output_dir / "annotation_template.jsonl"
    with open(template_path, "w") as f:
        for entry in annotations_template:
            f.write(json.dumps(entry) + "\n")

    # Generate HTML pages (batches of 25 pairs per page)
    batch_size = 25
    for batch_idx in range(0, len(annotations_template), batch_size):
        batch = annotations_template[batch_idx:batch_idx + batch_size]
        page_num = batch_idx // batch_size + 1

        html = _build_comparison_html(batch, page_num, pairs_dir)
        html_path = output_dir / f"comparison_page_{page_num:03d}.html"
        with open(html_path, "w") as f:
            f.write(html)

    # Generate annotator instructions
    instructions_path = output_dir / "ANNOTATOR_INSTRUCTIONS.md"
    with open(instructions_path, "w") as f:
        f.write(_build_instructions(len(annotations_template), output_dir))

    print(f"Generated {len(annotations_template)} comparison pairs")
    print(f"HTML pages: {output_dir}/comparison_page_*.html")
    print(f"Annotation template: {template_path}")
    print(f"Instructions: {instructions_path}")


def _build_comparison_html(batch: list[dict], page_num: int, pairs_dir: Path) -> str:
    """Build an HTML page for A/B comparison."""
    rows = ""
    for i, entry in enumerate(batch):
        pair_id = entry["pair_id"]
        prompt = entry["prompt"]
        left_path = pairs_dir / entry["video_left"]
        right_path = pairs_dir / entry["video_right"]

        rows += f"""
        <div class="pair" id="pair-{pair_id}">
            <div class="prompt"><strong>Prompt:</strong> {prompt}</div>
            <div class="videos">
                <div class="video-container">
                    <h3>Video A</h3>
                    <video controls loop muted playsinline width="400">
                        <source src="{left_path}" type="video/mp4">
                    </video>
                </div>
                <div class="video-container">
                    <h3>Video B</h3>
                    <video controls loop muted playsinline width="400">
                        <source src="{right_path}" type="video/mp4">
                    </video>
                </div>
            </div>
            <div class="buttons">
                <button onclick="select('{pair_id}', 'left')" id="btn-{pair_id}-left">A is better</button>
                <button onclick="select('{pair_id}', 'tie')" id="btn-{pair_id}-tie">Tie</button>
                <button onclick="select('{pair_id}', 'right')" id="btn-{pair_id}-right">B is better</button>
            </div>
            <div class="selected" id="label-{pair_id}"></div>
        </div>
        <hr>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>VLM-DPO Human Evaluation - Page {page_num}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
        .pair {{ margin: 30px 0; padding: 20px; background: #16213e; border-radius: 8px; }}
        .prompt {{ margin-bottom: 15px; font-size: 14px; color: #a0a0a0; }}
        .videos {{ display: flex; gap: 20px; justify-content: center; }}
        .video-container {{ text-align: center; }}
        .video-container h3 {{ margin: 5px 0; }}
        video {{ border-radius: 4px; }}
        .buttons {{ display: flex; gap: 10px; justify-content: center; margin-top: 15px; }}
        button {{ padding: 10px 24px; border: 2px solid #4a4a6a; border-radius: 6px; background: #0f3460; color: #e0e0e0; cursor: pointer; font-size: 14px; }}
        button:hover {{ background: #1a5276; }}
        button.active {{ background: #e94560; border-color: #e94560; font-weight: bold; }}
        .selected {{ text-align: center; margin-top: 8px; color: #e94560; font-weight: bold; }}
        h1 {{ color: #e94560; }}
        .progress {{ position: sticky; top: 0; background: #1a1a2e; padding: 10px; border-bottom: 2px solid #e94560; z-index: 100; }}
        .export-btn {{ padding: 12px 30px; background: #e94560; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }}
        .export-btn:hover {{ background: #c73e54; }}
    </style>
</head>
<body>
    <div class="progress">
        <h1>VLM-DPO Human Evaluation — Page {page_num}</h1>
        <p>Labeled: <span id="count">0</span> / {len(batch)} pairs</p>
        <button class="export-btn" onclick="exportResults()">Export Results</button>
    </div>

    {rows}

    <div style="text-align: center; margin: 40px 0;">
        <button class="export-btn" onclick="exportResults()">Export Results (JSON)</button>
    </div>

    <script>
        const results = {{}};
        let count = 0;

        function select(pairId, label) {{
            // Update button styles
            ['left', 'tie', 'right'].forEach(l => {{
                document.getElementById(`btn-${{pairId}}-${{l}}`).classList.remove('active');
            }});
            document.getElementById(`btn-${{pairId}}-${{label}}`).classList.add('active');

            // Track selection
            if (!results[pairId]) count++;
            results[pairId] = label;

            document.getElementById(`label-${{pairId}}`).textContent =
                label === 'left' ? 'Selected: A' : label === 'right' ? 'Selected: B' : 'Selected: Tie';
            document.getElementById('count').textContent = count;
        }}

        function exportResults() {{
            const data = JSON.stringify(results, null, 2);
            const blob = new Blob([data], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `annotations_page_{page_num}.json`;
            a.click();
        }}
    </script>
</body>
</html>"""


def _build_instructions(num_pairs: int, output_dir: Path) -> str:
    """Build annotator instruction document."""
    return f"""# Human Evaluation Instructions

## Task
You will compare {num_pairs} pairs of AI-generated videos. For each pair, choose which video better matches the text prompt.

## Criteria
Consider these aspects (in order of importance):
1. **Prompt adherence** — Does the video depict what the prompt describes?
2. **Visual quality** — Is the video clear, well-lit, and free of artifacts?
3. **Temporal consistency** — Are objects/characters consistent across frames?
4. **Motion naturalness** — Does the motion look realistic and smooth?

## How to annotate
1. Open the HTML pages in `{output_dir}/comparison_page_*.html`
2. For each pair, watch both videos (play them at least twice)
3. Click "A is better", "B is better", or "Tie"
4. When done with a page, click "Export Results" to download your annotations
5. Save all JSON files to `{output_dir}/`

## Important
- Video order (A/B) is randomized — there is no correct position
- If both videos are equally good or equally bad, choose "Tie"
- Focus on the prompt — a beautiful video that ignores the prompt should lose
- Take breaks every ~50 pairs to avoid fatigue

## After annotation
Combine all annotation JSON files and run:
```bash
python scripts/prepare_human_eval.py compute \\
    --vlm-scores data/exp1_vlm_agreement/metadata.jsonl \\
    --human-annotations {output_dir}/annotations.jsonl \\
    --output outputs/exp1/agreement_results.json
```
"""


def compute_agreement(
    vlm_scores_path: Path,
    human_annotations_path: Path,
    output_path: Path,
) -> None:
    """Compute agreement metrics between VLM and human preferences."""
    import numpy as np

    # Load VLM scores
    vlm_labels = {}
    with open(vlm_scores_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            # VLM winner is always the "winner" in metadata
            vlm_labels[entry["pair_id"]] = "winner"

    # Load human annotations
    human_labels = {}
    with open(human_annotations_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            pair_id = entry["pair_id"]
            label = entry.get("human_label")
            swapped = entry.get("swapped", False)

            if label is None:
                continue

            # Map back: if swapped, "left" in human UI = loser in VLM
            if label == "tie":
                human_labels[pair_id] = "tie"
            elif (label == "left" and not swapped) or (label == "right" and swapped):
                human_labels[pair_id] = "winner"
            else:
                human_labels[pair_id] = "loser"

    # Compute metrics
    common_ids = set(vlm_labels.keys()) & set(human_labels.keys())
    if not common_ids:
        print("ERROR: No overlapping pair IDs between VLM and human annotations.")
        return

    # Binary agreement (ignoring ties): does human agree with VLM winner?
    agree = 0
    disagree = 0
    ties = 0

    vlm_arr = []
    human_arr = []

    for pid in sorted(common_ids):
        h = human_labels[pid]
        if h == "tie":
            ties += 1
            continue
        elif h == "winner":
            agree += 1
            vlm_arr.append(1)
            human_arr.append(1)
        else:
            disagree += 1
            vlm_arr.append(1)
            human_arr.append(0)

    total_non_tie = agree + disagree
    agreement_rate = agree / total_non_tie if total_non_tie > 0 else 0

    # Cohen's kappa
    vlm_arr = np.array(vlm_arr)
    human_arr = np.array(human_arr)

    if len(vlm_arr) > 0:
        from sklearn.metrics import cohen_kappa_score
        # For kappa, we need both raters to have same label space
        # VLM always says "1" (winner), human says 1 (agree) or 0 (disagree)
        # Reframe: both rate each pair as "A better" or "B better"
        kappa = cohen_kappa_score(vlm_arr, human_arr)
    else:
        kappa = 0.0

    results = {
        "total_pairs": len(common_ids),
        "non_tie_pairs": total_non_tie,
        "ties": ties,
        "agreements": agree,
        "disagreements": disagree,
        "agreement_rate": round(agreement_rate, 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": _interpret_kappa(kappa),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("VLM vs Human Agreement Results")
    print("=" * 50)
    print(f"Total pairs evaluated:  {results['total_pairs']}")
    print(f"Non-tie pairs:          {results['non_tie_pairs']}")
    print(f"Ties:                   {results['ties']}")
    print(f"Agreements:             {results['agreements']}")
    print(f"Disagreements:          {results['disagreements']}")
    print(f"Agreement rate:         {results['agreement_rate']:.1%}")
    print(f"Cohen's kappa:          {results['cohens_kappa']:.4f}")
    print(f"Interpretation:         {results['interpretation']}")
    print(f"\nResults saved to: {output_path}")


def _interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's kappa value."""
    if kappa < 0:
        return "less than chance agreement"
    elif kappa < 0.2:
        return "slight agreement"
    elif kappa < 0.4:
        return "fair agreement"
    elif kappa < 0.6:
        return "moderate agreement"
    elif kappa < 0.8:
        return "substantial agreement"
    else:
        return "almost perfect agreement"


def main():
    parser = argparse.ArgumentParser(description="Human evaluation tools for Exp 1")
    subparsers = parser.add_subparsers(dest="command")

    # Generate subcommand
    gen = subparsers.add_parser("generate", help="Generate HTML comparison pages")
    gen.add_argument("--pairs-dir", required=True, help="Directory with generated pairs")
    gen.add_argument("--output-dir", required=True, help="Output directory for HTML pages")
    gen.add_argument("--num-pairs", type=int, default=None, help="Limit number of pairs")
    gen.add_argument("--seed", type=int, default=42, help="Seed for randomization")

    # Compute subcommand
    comp = subparsers.add_parser("compute", help="Compute VLM-human agreement")
    comp.add_argument("--vlm-scores", required=True, help="Path to VLM metadata.jsonl")
    comp.add_argument("--human-annotations", required=True, help="Path to human annotations JSONL")
    comp.add_argument("--output", required=True, help="Output path for results JSON")

    args = parser.parse_args()

    if args.command == "generate":
        generate_html_pages(
            pairs_dir=Path(args.pairs_dir),
            output_dir=Path(args.output_dir),
            num_pairs=args.num_pairs,
            seed=args.seed,
        )
    elif args.command == "compute":
        compute_agreement(
            vlm_scores_path=Path(args.vlm_scores),
            human_annotations_path=Path(args.human_annotations),
            output_path=Path(args.output),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
