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

mkdir -p "$LOG_DIR"

echo "=== VLM-DPO: Experiment 2 — Image DPO on Flux.2 ==="
echo "Timestamp: $TIMESTAMP"
echo "Logging to: $LOG_DIR"

# ------------------------------------------------------------------
# Step 1: Prepare prompt dataset (10K prompts for image DPO)
# ------------------------------------------------------------------
echo ""
echo "[1/5] Preparing prompt dataset..."
if [ ! -f "data/prompts.jsonl" ] || [ "$(wc -l < data/prompts.jsonl)" -lt 10000 ]; then
    python scripts/prepare_prompts.py \
        --output data/prompts.jsonl \
        --num-prompts 10000 \
        --seed 42 \
        2>&1 | tee "${LOG_DIR}/prepare_prompts_${TIMESTAMP}.log"
else
    echo "  Prompt dataset already exists ($(wc -l < data/prompts.jsonl) prompts). Skipping."
fi

# ------------------------------------------------------------------
# Step 2: Generate 10K image preference pairs (with checkpointing)
# ------------------------------------------------------------------
echo ""
echo "[2/5] Generating 10K image preference pairs with Flux.2..."
python scripts/generate_pairs.py \
    --config configs/exp2_image_dpo.yaml \
    --output-dir data/exp2_image_pairs \
    --num-pairs 10000 \
    --checkpoint-every 100 \
    --resume \
    2>&1 | tee "${LOG_DIR}/generate_${TIMESTAMP}.log"

echo "Image pair generation complete."

# ------------------------------------------------------------------
# Step 3: Train main image DPO model
# ------------------------------------------------------------------
echo ""
echo "[3/5] Training LoRA-DPO on Flux.2 (main run)..."
vlm-dpo train \
    --config configs/exp2_image_dpo.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/train_main_${TIMESTAMP}.log"

echo "Main training complete."

# ------------------------------------------------------------------
# Step 4: Evaluate
# ------------------------------------------------------------------
echo ""
echo "[4/5] Evaluating image DPO model..."
vlm-dpo evaluate \
    --config configs/exp2_image_dpo.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/eval_${TIMESTAMP}.log"

echo "Evaluation complete."

# ------------------------------------------------------------------
# Step 5: Verify checkpoint for Exp 4 transfer
# ------------------------------------------------------------------
echo ""
echo "[5/5] Verifying checkpoint for cross-modal transfer..."

if [ -d "${EXP_DIR}/best_checkpoint" ]; then
    echo "Best checkpoint found at: ${EXP_DIR}/best_checkpoint"
    echo "Ready for Experiment 4 (cross-modal transfer)."
else
    echo "WARNING: No best_checkpoint directory found."
    echo "Check training logs for issues. Exp 4 will not be able to run."
fi

echo ""
echo "Outputs:"
echo "  Prompts:        data/prompts.jsonl"
echo "  Image pairs:    data/exp2_image_pairs/"
echo "  Checkpoints:    ${EXP_DIR}/"
echo "  Metrics:        ${EXP_DIR}/metrics.json"
echo "  Logs:           ${LOG_DIR}/"
echo ""
echo "=== Experiment 2 complete! ==="
