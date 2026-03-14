"""
Human preference evaluation utilities.

Generates side-by-side comparison pages for collecting human preferences
and aggregates the results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def generate_comparison_html(
    pairs: list[dict[str, Any]],
    output_path: str | Path,
    modality: str = "video",
    title: str = "Human Preference Evaluation",
) -> None:
    """
    Generate an HTML page for side-by-side human preference evaluation.

    Args:
        pairs: List of dicts with "media_a_path", "media_b_path", "prompt", "pair_id".
        output_path: Path to write the HTML file.
        modality: "video" or "image".
        title: Page title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    media_tag = "video" if modality == "video" else "img"

    html_parts = [
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Inter', system-ui, sans-serif; background: #0f0f23; color: #e0e0e0; padding: 2rem; }}
    h1 {{ text-align: center; margin-bottom: 2rem; color: #7c8aff; }}
    .pair {{ background: #1a1a2e; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid #2a2a4a; }}
    .prompt {{ font-style: italic; color: #a0a0d0; margin-bottom: 1rem; padding: 0.5rem; background: #16162b; border-radius: 6px; }}
    .comparison {{ display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; }}
    .sample {{ flex: 1; min-width: 300px; text-align: center; }}
    .sample video, .sample img {{ max-width: 100%; border-radius: 8px; border: 2px solid transparent; cursor: pointer; transition: border-color 0.3s; }}
    .sample video:hover, .sample img:hover {{ border-color: #7c8aff; }}
    .sample.selected video, .sample.selected img {{ border-color: #4caf50; box-shadow: 0 0 20px rgba(76, 175, 80, 0.3); }}
    .label {{ margin-top: 0.5rem; font-weight: 600; color: #888; }}
    .btn-group {{ display: flex; gap: 1rem; justify-content: center; margin-top: 1rem; }}
    button {{ padding: 0.6rem 1.5rem; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; transition: all 0.2s; }}
    .btn-a {{ background: #3a5fcd; color: white; }}
    .btn-b {{ background: #cd3a5f; color: white; }}
    .btn-tie {{ background: #555; color: white; }}
    button:hover {{ transform: scale(1.05); filter: brightness(1.2); }}
    button.active {{ box-shadow: 0 0 15px rgba(255,255,255,0.3); }}
    #submit {{ display: block; margin: 2rem auto; padding: 1rem 3rem; background: #4caf50; color: white; font-size: 1.1rem; border: none; border-radius: 8px; cursor: pointer; }}
    #submit:hover {{ background: #45a049; }}
    #results {{ display: none; margin: 2rem auto; max-width: 600px; background: #1a1a2e; padding: 1.5rem; border-radius: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p style="text-align:center;color:#888;margin-bottom:2rem;">
    For each pair, select which {modality} is better (A, B, or Tie).
</p>
"""
    ]

    for pair in pairs:
        pair_id = pair["pair_id"]
        prompt = pair["prompt"]
        media_a = pair["media_a_path"]
        media_b = pair["media_b_path"]

        if modality == "video":
            media_html_a = f'<video controls loop muted><source src="{media_a}" type="video/mp4"></video>'
            media_html_b = f'<video controls loop muted><source src="{media_b}" type="video/mp4"></video>'
        else:
            media_html_a = f'<img src="{media_a}" alt="Sample A">'
            media_html_b = f'<img src="{media_b}" alt="Sample B">'

        html_parts.append(f"""
<div class="pair" id="pair-{pair_id}">
    <div class="prompt"><strong>Prompt:</strong> {prompt}</div>
    <div class="comparison">
        <div class="sample" id="{pair_id}-a">
            {media_html_a}
            <div class="label">Sample A</div>
        </div>
        <div class="sample" id="{pair_id}-b">
            {media_html_b}
            <div class="label">Sample B</div>
        </div>
    </div>
    <div class="btn-group">
        <button class="btn-a" onclick="select('{pair_id}', 'A', this)">A is Better</button>
        <button class="btn-tie" onclick="select('{pair_id}', 'tie', this)">Tie</button>
        <button class="btn-b" onclick="select('{pair_id}', 'B', this)">B is Better</button>
    </div>
</div>
""")

    html_parts.append("""
<button id="submit" onclick="submitResults()">Submit Preferences</button>
<div id="results">
    <h3 style="color:#7c8aff;margin-bottom:1rem;">Results Saved!</h3>
    <pre id="results-json" style="color:#aaa;white-space:pre-wrap;"></pre>
</div>

<script>
const preferences = {};

function select(pairId, choice, btn) {
    preferences[pairId] = choice;
    const group = btn.parentElement;
    group.querySelectorAll('button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

function submitResults() {
    const json = JSON.stringify(preferences, null, 2);
    document.getElementById('results-json').textContent = json;
    document.getElementById('results').style.display = 'block';

    // Download as file
    const blob = new Blob([json], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'human_preferences.json';
    a.click();
}
</script>
</body>
</html>""")

    html_content = "\n".join(html_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Human eval page generated: {output_path} ({len(pairs)} pairs)")


def aggregate_human_preferences(
    results_file: str | Path,
) -> dict[str, Any]:
    """
    Aggregate human preference results.

    Args:
        results_file: Path to the JSON file with human preferences.

    Returns:
        Dict with win rates, tie rate, and summary stats.
    """
    with open(results_file) as f:
        prefs = json.load(f)

    total = len(prefs)
    wins_a = sum(1 for v in prefs.values() if v == "A")
    wins_b = sum(1 for v in prefs.values() if v == "B")
    ties = sum(1 for v in prefs.values() if v == "tie")

    results = {
        "total_comparisons": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": wins_a / total if total > 0 else 0,
        "win_rate_b": wins_b / total if total > 0 else 0,
        "tie_rate": ties / total if total > 0 else 0,
    }

    logger.info(
        f"Human preferences: A={wins_a} ({results['win_rate_a']:.1%}), "
        f"B={wins_b} ({results['win_rate_b']:.1%}), "
        f"Tie={ties} ({results['tie_rate']:.1%})"
    )
    return results
