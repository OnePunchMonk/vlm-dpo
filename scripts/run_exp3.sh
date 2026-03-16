#!/bin/bash
# ============================================================
# Experiment 3: Video DPO on Wan2.1 (Main Result)
# ============================================================
# LoRA-DPO on Wan2.1-1.3B with 5K VLM preference video pairs.
# Includes ablations over beta, LoRA rank, and data scale.
# Includes baselines: SFT-only, random preference.
#
# Prerequisites: None (core experiment)
# Hardware: ~16 GB VRAM (Wan2.1) + ~10 GB (InternVL)
# Estimated time: ~20 hours generation + ~12 hours main training
#                 + ~50 hours ablations (can parallelize on multi-GPU)
# ============================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="outputs/exp3"
LOG_DIR="${EXP_DIR}/logs"
ABLATION_DIR="${EXP_DIR}/ablations"

mkdir -p "$LOG_DIR" "$ABLATION_DIR"

echo "=== VLM-DPO: Experiment 3 — Video DPO on Wan2.1 (Main Result) ==="
echo "Timestamp: $TIMESTAMP"
echo "Logging to: $LOG_DIR"

# ------------------------------------------------------------------
# Step 1: Generate 5K video preference pairs
# ------------------------------------------------------------------
echo ""
echo "[1/6] Generating 5K video preference pairs with Wan2.1..."
vlm-dpo generate \
    --config configs/exp3_video_dpo.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/generate_${TIMESTAMP}.log"

echo "Video pair generation complete."

# ------------------------------------------------------------------
# Step 2: Train main model (β=0.1, rank=16, 5K pairs)
# ------------------------------------------------------------------
echo ""
echo "[2/6] Training main LoRA-DPO on Wan2.1..."
vlm-dpo train \
    --config configs/exp3_video_dpo.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/train_main_${TIMESTAMP}.log"

echo "Main training complete."

# ------------------------------------------------------------------
# Step 3: Train baselines
# ------------------------------------------------------------------
echo ""
echo "[3/6] Training baselines..."

# Baseline A: SFT-only (train on winner videos only, no preference signal)
# Use beta=0 to effectively disable the DPO preference loss component
echo "  --- Baseline: SFT-only ---"
vlm-dpo train \
    --config configs/exp3_video_dpo.yaml \
    --overrides \
        "dpo.beta=0.0" \
        "training.wandb_run_name=exp3-baseline-sft-only" \
    2>&1 | tee "${LOG_DIR}/train_sft_only_${TIMESTAMP}.log"

# Baseline B: Random preference (shuffled winner/loser labels)
# Generate new pairs with random scoring to create shuffled preferences
echo "  --- Baseline: Random preference ---"
vlm-dpo train \
    --config configs/exp3_video_dpo.yaml \
    --overrides \
        "dpo.label_smoothing=0.5" \
        "training.wandb_run_name=exp3-baseline-random-pref" \
    2>&1 | tee "${LOG_DIR}/train_random_pref_${TIMESTAMP}.log"

echo "Baselines complete."

# ------------------------------------------------------------------
# Step 4: Ablation — DPO temperature (β)
# ------------------------------------------------------------------
echo ""
echo "[4/6] Running β ablation..."

for beta in 0.01 0.05 0.1 0.2 0.5; do
    echo "  --- Beta=${beta} ---"
    vlm-dpo train \
        --config configs/exp3_video_dpo.yaml \
        --overrides \
            "dpo.beta=${beta}" \
            "training.wandb_run_name=exp3-ablation-beta-${beta}" \
        2>&1 | tee "${LOG_DIR}/train_beta_${beta}_${TIMESTAMP}.log"
done

echo "Beta ablation complete."

# ------------------------------------------------------------------
# Step 5: Ablation — LoRA rank
# ------------------------------------------------------------------
echo ""
echo "[5/6] Running LoRA rank ablation..."

for rank in 4 8 16 32 64; do
    echo "  --- Rank=${rank} ---"
    vlm-dpo train \
        --config configs/exp3_video_dpo.yaml \
        --overrides \
            "lora.rank=${rank}" \
            "lora.alpha=$((rank * 2))" \
            "training.wandb_run_name=exp3-ablation-rank-${rank}" \
        2>&1 | tee "${LOG_DIR}/train_rank_${rank}_${TIMESTAMP}.log"
done

echo "Rank ablation complete."

# ------------------------------------------------------------------
# Step 5b: Ablation — Data scale
# ------------------------------------------------------------------
echo ""
echo "  Running data scale ablation..."

for num_pairs in 500 1000 2000 5000; do
    echo "  --- NumPairs=${num_pairs} ---"
    vlm-dpo train \
        --config configs/exp3_video_dpo.yaml \
        --overrides \
            "data.num_pairs=${num_pairs}" \
            "training.wandb_run_name=exp3-ablation-pairs-${num_pairs}" \
        2>&1 | tee "${LOG_DIR}/train_pairs_${num_pairs}_${TIMESTAMP}.log"
done

echo "Data scale ablation complete."

# ------------------------------------------------------------------
# Step 6: Evaluate all models
# ------------------------------------------------------------------
echo ""
echo "[6/6] Evaluating main model and ablations..."

# Evaluate main model
vlm-dpo evaluate \
    --config configs/exp3_video_dpo.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/eval_main_${TIMESTAMP}.log"

echo ""
echo "Outputs:"
echo "  Video pairs:       data/exp3_video_pairs/"
echo "  Main checkpoint:   ${EXP_DIR}/best_checkpoint/"
echo "  Ablation results:  ${ABLATION_DIR}/"
echo "  Metrics:           ${EXP_DIR}/metrics.json"
echo "  Logs:              ${LOG_DIR}/"
echo ""
echo "=== Experiment 3 complete! ==="
