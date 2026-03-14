#!/bin/bash
# ============================================================
# Experiment 5: Multi-Aspect Reward Decomposition
# ============================================================
# Compares holistic vs decomposed multi-aspect DPO.
# Ablates reward weights α_p, α_t, α_v, α_m.

set -euo pipefail

echo "=== VLM-DPO: Experiment 5 — Multi-Aspect Reward Decomposition ==="

# Step 1: Holistic baseline
echo "[1/3] Training holistic DPO baseline..."
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides "scoring.strategy=holistic" "training.wandb_run_name=exp5-holistic" \
    -v

# Step 2: Multi-aspect with different weight configs
echo "[2/3] Running multi-aspect ablations..."

# Equal weights
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.25" \
        "scoring.reward_weights.temporal_consistency=0.25" \
        "scoring.reward_weights.visual_quality=0.25" \
        "scoring.reward_weights.motion_naturalness=0.25" \
        "training.wandb_run_name=exp5-equal-weights"

# Temporal-heavy
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.2" \
        "scoring.reward_weights.temporal_consistency=0.5" \
        "scoring.reward_weights.visual_quality=0.15" \
        "scoring.reward_weights.motion_naturalness=0.15" \
        "training.wandb_run_name=exp5-temporal-heavy"

# Prompt-heavy
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.5" \
        "scoring.reward_weights.temporal_consistency=0.2" \
        "scoring.reward_weights.visual_quality=0.15" \
        "scoring.reward_weights.motion_naturalness=0.15" \
        "training.wandb_run_name=exp5-prompt-heavy"

# Step 3: Evaluate all
echo "[3/3] Evaluating..."
vlm-dpo evaluate \
    --config configs/exp5_multi_aspect.yaml \
    -v

echo "=== Experiment 5 complete! Results in outputs/exp5/ ==="
