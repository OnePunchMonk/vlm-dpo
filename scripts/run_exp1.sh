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
# Step 1: Generate 500 video pairs with Wan2.1
# ------------------------------------------------------------------
echo ""
echo "[1/5] Generating 500 video pairs..."
vlm-dpo generate \
    --config configs/exp1_vlm_agreement.yaml \
    -v \
    2>&1 | tee "${LOG_DIR}/generate_${TIMESTAMP}.log"

echo "Video pair generation complete."

# ------------------------------------------------------------------
# Step 2: Score with all three strategies (for ablation)
# ------------------------------------------------------------------
echo ""
echo "[2/5] Scoring with all three VLM strategies..."

for strategy in holistic multi_aspect cot; do
    echo ""
    echo "  --- Scoring strategy: ${strategy} ---"
    vlm-dpo generate \
        --config configs/exp1_vlm_agreement.yaml \
        --overrides \
            "scoring.strategy=${strategy}" \
            "data.output_dir=data/exp1_${strategy}" \
        2>&1 | tee "${LOG_DIR}/score_${strategy}_${TIMESTAMP}.log"
    echo "  Strategy ${strategy} complete."
done

echo "All scoring strategies complete."

# ------------------------------------------------------------------
# Step 3: Generate HTML for human evaluation
# ------------------------------------------------------------------
echo ""
echo "[3/5] Generating human evaluation interface..."

# The evaluate command with human_preference metric will generate HTML
# comparisons for annotation. The HTML files go to human_eval_dir.
vlm-dpo evaluate \
    --config configs/exp1_vlm_agreement.yaml \
    --overrides "eval.eval_output_dir=${HUMAN_EVAL_DIR}" \
    -v \
    2>&1 | tee "${LOG_DIR}/human_eval_gen_${TIMESTAMP}.log"

echo "Human evaluation interface generated at: ${HUMAN_EVAL_DIR}"
echo ""
echo "============================================================"
echo "  ACTION REQUIRED: Collect human annotations"
echo "  1. Open ${HUMAN_EVAL_DIR}/*.html in a browser"
echo "  2. Have 3 annotators label each pair (A wins / B wins / Tie)"
echo "  3. Save results to data/human_annotations.jsonl"
echo "  4. Then run step 4 below"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# Step 4: Compute agreement metrics (run after human annotation)
# ------------------------------------------------------------------
# Uncomment this block after collecting human annotations:

# HUMAN_ANNOTATIONS="data/human_annotations.jsonl"
# if [ ! -f "$HUMAN_ANNOTATIONS" ]; then
#     echo "ERROR: Human annotations not found at $HUMAN_ANNOTATIONS"
#     echo "Complete human annotation before running this step."
#     exit 1
# fi
#
# echo "[4/5] Computing Cohen's kappa and agreement metrics..."
#
# for strategy in holistic multi_aspect cot; do
#     echo "  Computing agreement for strategy: ${strategy}"
#     vlm-dpo evaluate \
#         --config configs/exp1_vlm_agreement.yaml \
#         --overrides \
#             "scoring.strategy=${strategy}" \
#             "eval.eval_output_dir=${EXP_DIR}/agreement_${strategy}" \
#         -v \
#         2>&1 | tee "${LOG_DIR}/agreement_${strategy}_${TIMESTAMP}.log"
# done
#
# echo "Agreement metrics computed."

# ------------------------------------------------------------------
# Step 5: Summarize results
# ------------------------------------------------------------------
echo ""
echo "[5/5] Experiment 1 summary"
echo ""
echo "Outputs:"
echo "  VLM scores (holistic):     data/exp1_holistic/metadata.jsonl"
echo "  VLM scores (multi_aspect): data/exp1_multi_aspect/metadata.jsonl"
echo "  VLM scores (cot):          data/exp1_cot/metadata.jsonl"
echo "  Human eval interface:      ${HUMAN_EVAL_DIR}/"
echo "  Logs:                      ${LOG_DIR}/"
echo ""
echo "=== Experiment 1 complete! ==="
