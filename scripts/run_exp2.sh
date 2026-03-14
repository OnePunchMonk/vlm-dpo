#!/bin/bash
# ============================================================
# Experiment 2: Image DPO on Flux.2
# ============================================================
# LoRA-DPO on Flux.2 with 10K InternVL-U preference image pairs.

set -euo pipefail

echo "=== VLM-DPO: Experiment 2 — Image DPO on Flux.2 ==="

# Step 1: Generate image preference pairs
echo "[1/3] Generating 10K image pairs with Flux.2..."
vlm-dpo generate \
    --config configs/exp2_image_dpo.yaml \
    -v

# Step 2: Train with DPO
echo "[2/3] Training LoRA-DPO on Flux.2..."
vlm-dpo train \
    --config configs/exp2_image_dpo.yaml \
    -v

# Step 3: Evaluate
echo "[3/3] Evaluating..."
vlm-dpo evaluate \
    --config configs/exp2_image_dpo.yaml \
    -v

echo "=== Experiment 2 complete! Results in outputs/exp2/ ==="
