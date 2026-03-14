#!/bin/bash
# ============================================================
# Experiment 4: Cross-Modal Transfer (Image → Video)
# ============================================================
# Tests whether image DPO LoRA weights initialize video DPO better.
# Requires Exp 2 to have been run first (source LoRA).

set -euo pipefail

echo "=== VLM-DPO: Experiment 4 — Cross-Modal Transfer ==="

# Verify Exp 2 checkpoint exists
if [ ! -d "outputs/exp2/best_checkpoint" ]; then
    echo "ERROR: Exp 2 checkpoint not found. Run Exp 2 first."
    exit 1
fi

# Step 1: Transfer LoRA and train
echo "[1/2] Training video DPO with image LoRA initialization..."
vlm-dpo train \
    --config configs/exp4_cross_modal.yaml \
    -v

# Step 2: Evaluate and compare with Exp 3
echo "[2/2] Evaluating..."
vlm-dpo evaluate \
    --config configs/exp4_cross_modal.yaml \
    -v

echo "=== Experiment 4 complete! Results in outputs/exp4/ ==="
