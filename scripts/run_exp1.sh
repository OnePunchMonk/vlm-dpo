#!/bin/bash
# ============================================================
# Experiment 1: VLM Preference Quality
# ============================================================
# Benchmarks InternVL-U as a preference oracle against humans.
# Ablates scoring strategies: holistic, multi-aspect, CoT.

set -euo pipefail

echo "=== VLM-DPO: Experiment 1 — VLM Preference Quality ==="

# Step 1: Generate video pairs
echo "[1/3] Generating 500 video pairs..."
vlm-dpo generate \
    --config configs/exp1_vlm_agreement.yaml \
    -v

# Step 2: Score with all three strategies (for ablation)
echo "[2/3] Scoring with holistic, multi-aspect, and CoT strategies..."
for strategy in holistic multi_aspect cot; do
    echo "  Scoring with strategy: $strategy"
    vlm-dpo generate \
        --config configs/exp1_vlm_agreement.yaml \
        --overrides "scoring.strategy=$strategy" "data.output_dir=data/exp1_${strategy}"
done

# Step 3: Compute agreement metrics
echo "[3/3] Computing Cohen's kappa..."
vlm-dpo evaluate \
    --config configs/exp1_vlm_agreement.yaml \
    -v

echo "=== Experiment 1 complete! Results in outputs/exp1/ ==="
