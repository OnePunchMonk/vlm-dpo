#!/bin/bash
# ============================================================
# Experiment 4: Cross-Modal Transfer (Image → Video)
# ============================================================
# Tests whether image DPO LoRA weights initialize video DPO better.
# Requires Exp 2 to have been run first (source LoRA from Flux.2).
# Compares against Exp 3 from-scratch baseline.
#
# Prerequisites: Experiment 2 (image DPO checkpoint)
# Hardware: ~16 GB VRAM (Wan2.1 training)
# Estimated time: ~12 hours main + ~24 hours ablations
# ============================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="outputs/exp4"
LOG_DIR="${EXP_DIR}/logs"
ABLATION_DIR="${EXP_DIR}/ablations"

mkdir -p "$LOG_DIR" "$ABLATION_DIR"

echo "=== VLM-DPO: Experiment 4 — Cross-Modal Transfer ==="
echo "Timestamp: $TIMESTAMP"
echo "Logging to: $LOG_DIR"

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------
SOURCE_LORA="outputs/exp2/best_checkpoint"
if [ ! -d "$SOURCE_LORA" ]; then
    echo "ERROR: Exp 2 checkpoint not found at $SOURCE_LORA"
    echo "Run Experiment 2 first: bash scripts/run_exp2.sh"
    exit 1
fi
echo "Source LoRA found: $SOURCE_LORA"

EXP3_BASELINE="outputs/exp3/metrics.json"
if [ ! -f "$EXP3_BASELINE" ]; then
    echo "WARNING: Exp 3 baseline metrics not found at $EXP3_BASELINE"
    echo "Run Experiment 3 first for comparison. Continuing without baseline."
fi

# ------------------------------------------------------------------
# Step 1: Train with cross-modal transfer (main config)
# ------------------------------------------------------------------
echo ""
echo "[1/4] Training video DPO with image LoRA initialization..."
echo "  Transfer strategy: name_match"
echo "  Freeze transferred layers for 200 steps"
vlm-dpo train \
    --config configs/exp4_cross_modal.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/train_transfer_main_${TIMESTAMP}.log"

echo "Main transfer training complete."

# ------------------------------------------------------------------
# Step 2: Ablation — Mapping strategy
# ------------------------------------------------------------------
echo ""
echo "[2/4] Running mapping strategy ablation..."

for strategy in name_match position_match; do
    echo "  --- Strategy: ${strategy} ---"
    vlm-dpo train \
        --config configs/exp4_cross_modal.yaml \
        --overrides \
            "transfer.mapping_strategy=${strategy}" \
            "training.wandb_run_name=exp4-strategy-${strategy}" \
        2>&1 | tee "${LOG_DIR}/train_strategy_${strategy}_${TIMESTAMP}.log"
done

echo "Mapping strategy ablation complete."

# ------------------------------------------------------------------
# Step 3: Ablation — Freeze duration
# ------------------------------------------------------------------
echo ""
echo "[3/4] Running freeze duration ablation..."

for freeze_steps in 0 100 200 500; do
    echo "  --- Freeze steps: ${freeze_steps} ---"
    vlm-dpo train \
        --config configs/exp4_cross_modal.yaml \
        --overrides \
            "transfer.freeze_transferred_steps=${freeze_steps}" \
            "training.wandb_run_name=exp4-freeze-${freeze_steps}" \
        2>&1 | tee "${LOG_DIR}/train_freeze_${freeze_steps}_${TIMESTAMP}.log"
done

echo "Freeze duration ablation complete."

# ------------------------------------------------------------------
# Step 4: Evaluate and compare with Exp 3
# ------------------------------------------------------------------
echo ""
echo "[4/4] Evaluating transfer model..."
vlm-dpo evaluate \
    --config configs/exp4_cross_modal.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/eval_${TIMESTAMP}.log"

# Print comparison if Exp 3 baseline exists
if [ -f "$EXP3_BASELINE" ]; then
    echo ""
    echo "--- Comparison with Exp 3 (from-scratch) ---"
    echo "Exp 3 baseline:"
    cat "$EXP3_BASELINE"
    echo ""
    echo "Exp 4 transfer:"
    cat "${EXP_DIR}/metrics.json" 2>/dev/null || echo "  (metrics not yet generated)"
fi

echo ""
echo "Outputs:"
echo "  Transfer checkpoint:  ${EXP_DIR}/best_checkpoint/"
echo "  Ablation results:     ${ABLATION_DIR}/"
echo "  Metrics:              ${EXP_DIR}/metrics.json"
echo "  Transfer analysis:    ${EXP_DIR}/transfer_analysis.json"
echo "  Logs:                 ${LOG_DIR}/"
echo ""
echo "=== Experiment 4 complete! ==="
