#!/bin/bash
# ============================================================
# Experiment 3: Video DPO on Wan2.1 (Main Result)
# ============================================================
# LoRA-DPO on Wan2.1-1.3B with 5K VLM preference video pairs.
# Includes ablations over beta, LoRA rank, and num_pairs.

set -euo pipefail

echo "=== VLM-DPO: Experiment 3 — Video DPO on Wan2.1 (Main Result) ==="

# Step 1: Generate video preference pairs
echo "[1/4] Generating 5K video pairs with Wan2.1..."
vlm-dpo generate \
    --config configs/exp3_video_dpo.yaml \
    -v

# Step 2: Train main model
echo "[2/4] Training LoRA-DPO on Wan2.1..."
vlm-dpo train \
    --config configs/exp3_video_dpo.yaml \
    -v

# Step 3: Ablations
echo "[3/4] Running ablations..."

# Beta ablation
for beta in 0.01 0.05 0.1 0.2 0.5; do
    echo "  Beta=$beta"
    vlm-dpo train \
        --config configs/exp3_video_dpo.yaml \
        --overrides "dpo.beta=$beta" "training.wandb_run_name=exp3-beta-$beta"
done

# LoRA rank ablation
for rank in 4 8 16 32 64; do
    echo "  LoRA rank=$rank"
    vlm-dpo train \
        --config configs/exp3_video_dpo.yaml \
        --overrides "lora.rank=$rank" "training.wandb_run_name=exp3-rank-$rank"
done

# Step 4: Evaluate
echo "[4/4] Evaluating..."
vlm-dpo evaluate \
    --config configs/exp3_video_dpo.yaml \
    -v

echo "=== Experiment 3 complete! Results in outputs/exp3/ ==="
