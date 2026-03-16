#!/bin/bash
# ============================================================
# Experiment 5: Multi-Aspect Reward Decomposition
# ============================================================
# Compares holistic vs decomposed multi-aspect DPO.
# Ablates reward weights α_p, α_t, α_v, α_m to identify
# which quality dimension drives the most improvement.
#
# Prerequisites: Exp 3 data (5K pairs with multi-aspect scores)
# Hardware: ~16 GB VRAM (Wan2.1 training)
# Estimated time: ~8 hours per config × 8 configs = ~64 hours
#                 (can parallelize on multi-GPU)
# ============================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="outputs/exp5"
LOG_DIR="${EXP_DIR}/logs"
ABLATION_DIR="${EXP_DIR}/ablations"

mkdir -p "$LOG_DIR" "$ABLATION_DIR"

echo "=== VLM-DPO: Experiment 5 — Multi-Aspect Reward Decomposition ==="
echo "Timestamp: $TIMESTAMP"
echo "Logging to: $LOG_DIR"

# ------------------------------------------------------------------
# Step 1: Holistic baseline (single score, no decomposition)
# ------------------------------------------------------------------
echo ""
echo "[1/4] Training holistic DPO baseline..."
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.strategy=holistic" \
        "training.wandb_run_name=exp5-holistic" \
    -v \
    2>&1 | tee "${LOG_DIR}/train_holistic_${TIMESTAMP}.log"

echo "Holistic baseline complete."

# ------------------------------------------------------------------
# Step 2: Single-dimension isolation runs
# ------------------------------------------------------------------
echo ""
echo "[2/4] Running single-dimension isolation..."

# Prompt adherence only
echo "  --- Prompt adherence only ---"
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=1.0" \
        "scoring.reward_weights.temporal_consistency=0.0" \
        "scoring.reward_weights.visual_quality=0.0" \
        "scoring.reward_weights.motion_naturalness=0.0" \
        "training.wandb_run_name=exp5-prompt-only" \
    2>&1 | tee "${LOG_DIR}/train_prompt_only_${TIMESTAMP}.log"

# Temporal consistency only
echo "  --- Temporal consistency only ---"
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.0" \
        "scoring.reward_weights.temporal_consistency=1.0" \
        "scoring.reward_weights.visual_quality=0.0" \
        "scoring.reward_weights.motion_naturalness=0.0" \
        "training.wandb_run_name=exp5-temporal-only" \
    2>&1 | tee "${LOG_DIR}/train_temporal_only_${TIMESTAMP}.log"

# Visual quality only
echo "  --- Visual quality only ---"
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.0" \
        "scoring.reward_weights.temporal_consistency=0.0" \
        "scoring.reward_weights.visual_quality=1.0" \
        "scoring.reward_weights.motion_naturalness=0.0" \
        "training.wandb_run_name=exp5-visual-only" \
    2>&1 | tee "${LOG_DIR}/train_visual_only_${TIMESTAMP}.log"

# Motion naturalness only
echo "  --- Motion naturalness only ---"
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.0" \
        "scoring.reward_weights.temporal_consistency=0.0" \
        "scoring.reward_weights.visual_quality=0.0" \
        "scoring.reward_weights.motion_naturalness=1.0" \
        "training.wandb_run_name=exp5-motion-only" \
    2>&1 | tee "${LOG_DIR}/train_motion_only_${TIMESTAMP}.log"

echo "Single-dimension isolation complete."

# ------------------------------------------------------------------
# Step 3: Combined weighting strategies
# ------------------------------------------------------------------
echo ""
echo "[3/4] Running combined weighting strategies..."

# Equal weights (uniform)
echo "  --- Equal weights (0.25 each) ---"
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.25" \
        "scoring.reward_weights.temporal_consistency=0.25" \
        "scoring.reward_weights.visual_quality=0.25" \
        "scoring.reward_weights.motion_naturalness=0.25" \
        "training.wandb_run_name=exp5-equal-weights" \
    2>&1 | tee "${LOG_DIR}/train_equal_weights_${TIMESTAMP}.log"

# Balanced weights (our proposed default)
echo "  --- Balanced weights (0.3, 0.3, 0.2, 0.2) ---"
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.3" \
        "scoring.reward_weights.temporal_consistency=0.3" \
        "scoring.reward_weights.visual_quality=0.2" \
        "scoring.reward_weights.motion_naturalness=0.2" \
        "training.wandb_run_name=exp5-balanced-weights" \
    2>&1 | tee "${LOG_DIR}/train_balanced_weights_${TIMESTAMP}.log"

# Temporal-heavy (hypothesis: temporal consistency matters most for video)
echo "  --- Temporal-heavy weights (0.2, 0.5, 0.15, 0.15) ---"
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.2" \
        "scoring.reward_weights.temporal_consistency=0.5" \
        "scoring.reward_weights.visual_quality=0.15" \
        "scoring.reward_weights.motion_naturalness=0.15" \
        "training.wandb_run_name=exp5-temporal-heavy" \
    2>&1 | tee "${LOG_DIR}/train_temporal_heavy_${TIMESTAMP}.log"

# Prompt-heavy (hypothesis: prompt adherence is the primary user need)
echo "  --- Prompt-heavy weights (0.5, 0.2, 0.15, 0.15) ---"
vlm-dpo train \
    --config configs/exp5_multi_aspect.yaml \
    --overrides \
        "scoring.reward_weights.prompt_adherence=0.5" \
        "scoring.reward_weights.temporal_consistency=0.2" \
        "scoring.reward_weights.visual_quality=0.15" \
        "scoring.reward_weights.motion_naturalness=0.15" \
        "training.wandb_run_name=exp5-prompt-heavy" \
    2>&1 | tee "${LOG_DIR}/train_prompt_heavy_${TIMESTAMP}.log"

echo "Combined weighting strategies complete."

# ------------------------------------------------------------------
# Step 4: Evaluate all configurations
# ------------------------------------------------------------------
echo ""
echo "[4/4] Evaluating all configurations..."
vlm-dpo evaluate \
    --config configs/exp5_multi_aspect.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/eval_${TIMESTAMP}.log"

echo ""
echo "Outputs:"
echo "  Ablation results:  ${ABLATION_DIR}/"
echo "  Metrics:           ${EXP_DIR}/metrics.json"
echo "  Logs:              ${LOG_DIR}/"
echo ""
echo "Configurations trained:"
echo "  1. Holistic baseline (single score)"
echo "  2. Prompt adherence only"
echo "  3. Temporal consistency only"
echo "  4. Visual quality only"
echo "  5. Motion naturalness only"
echo "  6. Equal weights (0.25, 0.25, 0.25, 0.25)"
echo "  7. Balanced weights (0.3, 0.3, 0.2, 0.2)"
echo "  8. Temporal-heavy (0.2, 0.5, 0.15, 0.15)"
echo "  9. Prompt-heavy (0.5, 0.2, 0.15, 0.15)"
echo ""
echo "=== Experiment 5 complete! ==="
