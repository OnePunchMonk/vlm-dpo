#!/bin/bash
# ============================================================
# Experiment 1: VLM Preference Quality
# ============================================================
# Benchmarks InternVL-U as a preference oracle against humans.
# Ablates scoring strategies: holistic, multi-aspect, CoT.
#
# Prerequisites: None (first experiment to run)
# Hardware: ~16 GB VRAM (Wan2.1 generation) + ~10 GB (InternVL scoring)
# Estimated time: ~3-4 hours generation + human annotation time
# ============================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="outputs/exp1"
LOG_DIR="${EXP_DIR}/logs"
HUMAN_EVAL_DIR="${EXP_DIR}/human_eval"

mkdir -p "$LOG_DIR" "$HUMAN_EVAL_DIR"

echo "=== VLM-DPO: Experiment 1 — VLM Preference Quality ==="
echo "Timestamp: $TIMESTAMP"
echo "Logging to: $LOG_DIR"

# ------------------------------------------------------------------
# Step 1: Prepare prompt dataset (500 diverse prompts)
# ------------------------------------------------------------------
echo ""
echo "[1/6] Preparing prompt dataset..."
if [ ! -f "data/prompts.jsonl" ] || [ "$(wc -l < data/prompts.jsonl)" -lt 500 ]; then
    python scripts/prepare_prompts.py \
        --output data/prompts.jsonl \
        --num-prompts 500 \
        --seed 42 \
        2>&1 | tee "${LOG_DIR}/prepare_prompts_${TIMESTAMP}.log"
else
    echo "  Prompt dataset already exists ($(wc -l < data/prompts.jsonl) prompts). Skipping."
fi

# ------------------------------------------------------------------
# Step 2: Generate 500 video pairs with checkpointing
# ------------------------------------------------------------------
echo ""
echo "[2/6] Generating 500 video pairs (with resume support)..."
python scripts/generate_pairs.py \
    --config configs/exp1_vlm_agreement.yaml \
    --output-dir data/exp1_vlm_agreement \
    --num-pairs 500 \
    --checkpoint-every 25 \
    --resume \
    2>&1 | tee "${LOG_DIR}/generate_${TIMESTAMP}.log"

echo "Video pair generation complete."

# ------------------------------------------------------------------
# Step 3: Re-score with all three strategies (for ablation)
# ------------------------------------------------------------------
echo ""
echo "[3/6] Scoring with all three VLM strategies..."

for strategy in holistic multi_aspect cot; do
    echo ""
    echo "  --- Scoring strategy: ${strategy} ---"
    python scripts/generate_pairs.py \
        --config configs/exp1_vlm_agreement.yaml \
        --output-dir "data/exp1_${strategy}" \
        --num-pairs 500 \
        --scoring-strategy "${strategy}" \
        --resume \
        2>&1 | tee "${LOG_DIR}/score_${strategy}_${TIMESTAMP}.log"
    echo "  Strategy ${strategy} complete."
done

echo "All scoring strategies complete."

# ------------------------------------------------------------------
# Step 4: Generate HTML for human evaluation
# ------------------------------------------------------------------
echo ""
echo "[4/6] Generating human evaluation interface..."
python scripts/prepare_human_eval.py generate \
    --pairs-dir data/exp1_vlm_agreement \
    --output-dir "${HUMAN_EVAL_DIR}" \
    --num-pairs 500 \
    2>&1 | tee "${LOG_DIR}/human_eval_gen_${TIMESTAMP}.log"

echo ""
echo "============================================================"
echo "  ACTION REQUIRED: Collect human annotations"
echo "  1. Open ${HUMAN_EVAL_DIR}/comparison_page_*.html in a browser"
echo "  2. Have 3 annotators label each pair (A wins / B wins / Tie)"
echo "  3. Export annotations from each page (click Export button)"
echo "  4. Combine into: ${HUMAN_EVAL_DIR}/annotations.jsonl"
echo "  5. Then run: bash scripts/run_exp1_agreement.sh"
echo "============================================================"

# ------------------------------------------------------------------
# Step 5: Compute agreement (run after human annotation)
# ------------------------------------------------------------------
# This step is split into a separate script since it requires
# human annotations to be collected first.

# ------------------------------------------------------------------
# Step 6: Summary
# ------------------------------------------------------------------
echo ""
echo "[6/6] Experiment 1 (Phase 1) summary"
echo ""
echo "Outputs:"
echo "  Prompts:                   data/prompts.jsonl"
echo "  Video pairs:               data/exp1_vlm_agreement/pairs/"
echo "  VLM scores (holistic):     data/exp1_holistic/metadata.jsonl"
echo "  VLM scores (multi_aspect): data/exp1_multi_aspect/metadata.jsonl"
echo "  VLM scores (cot):          data/exp1_cot/metadata.jsonl"
echo "  Human eval HTML:           ${HUMAN_EVAL_DIR}/"
echo "  Logs:                      ${LOG_DIR}/"
echo ""
echo "=== Experiment 1 (Phase 1) complete! ==="
echo "=== Run scripts/run_exp1_agreement.sh after collecting human annotations ==="
