#!/bin/bash
# ============================================================
# Experiment 2: Image DPO on Flux.2
# ============================================================
# LoRA-DPO on Flux.2-dev with 10K InternVL-U preference image pairs.
# This is a prerequisite for Experiment 4 (cross-modal transfer).
#
# Prerequisites: None (can run in parallel with Exp 1/3)
# Hardware: ~20 GB VRAM (Flux.2 FP8) + ~10 GB (InternVL scoring)
# Estimated time: ~14 hours generation + ~6-10 hours training
# ============================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="outputs/exp2"
LOG_DIR="${EXP_DIR}/logs"
CHECKPOINT_DIR="${EXP_DIR}/checkpoints"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "=== VLM-DPO: Experiment 2 — Image DPO on Flux.2 ==="
echo "Timestamp: $TIMESTAMP"
echo "Logging to: $LOG_DIR"

# ------------------------------------------------------------------
# Step 1: Generate 10K image preference pairs
# ------------------------------------------------------------------
echo ""
echo "[1/4] Generating 10K image preference pairs with Flux.2..."
vlm-dpo generate \
    --config configs/exp2_image_dpo.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/generate_${TIMESTAMP}.log"

echo "Image pair generation complete."

# ------------------------------------------------------------------
# Step 2: Train main image DPO model
# ------------------------------------------------------------------
echo ""
echo "[2/4] Training LoRA-DPO on Flux.2 (main run)..."
vlm-dpo train \
    --config configs/exp2_image_dpo.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/train_main_${TIMESTAMP}.log"

echo "Main training complete."

# ------------------------------------------------------------------
# Step 3: Evaluate
# ------------------------------------------------------------------
echo ""
echo "[3/4] Evaluating image DPO model..."
vlm-dpo evaluate \
    --config configs/exp2_image_dpo.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/eval_${TIMESTAMP}.log"

echo "Evaluation complete."

# ------------------------------------------------------------------
# Step 4: Verify checkpoint for Exp 4 transfer
# ------------------------------------------------------------------
echo ""
echo "[4/4] Verifying checkpoint for cross-modal transfer..."

if [ -d "${EXP_DIR}/best_checkpoint" ]; then
    echo "Best checkpoint found at: ${EXP_DIR}/best_checkpoint"
    echo "Ready for Experiment 4 (cross-modal transfer)."
else
    echo "WARNING: No best_checkpoint directory found."
    echo "Check training logs for issues. Exp 4 will not be able to run."
fi

echo ""
echo "Outputs:"
echo "  Image pairs:    data/exp2_image_pairs/"
echo "  Checkpoints:    ${EXP_DIR}/"
echo "  Metrics:        ${EXP_DIR}/metrics.json"
echo "  Logs:           ${LOG_DIR}/"
echo ""
echo "=== Experiment 2 complete! ==="
