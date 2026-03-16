#!/bin/bash
# ============================================================
# Experiment 1 (Phase 2): Compute VLM-Human Agreement
# ============================================================
# Run AFTER collecting human annotations from the HTML pages.
# Computes Cohen's kappa for each scoring strategy.
#
# Prerequisites: Human annotations collected in outputs/exp1/human_eval/annotations.jsonl
# ============================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="outputs/exp1"
LOG_DIR="${EXP_DIR}/logs"
HUMAN_EVAL_DIR="${EXP_DIR}/human_eval"
ANNOTATIONS="${HUMAN_EVAL_DIR}/annotations.jsonl"

echo "=== VLM-DPO: Experiment 1 (Phase 2) — Agreement Computation ==="

# Check for annotations
if [ ! -f "$ANNOTATIONS" ]; then
    echo "ERROR: Human annotations not found at: $ANNOTATIONS"
    echo ""
    echo "To create annotations:"
    echo "  1. Open ${HUMAN_EVAL_DIR}/comparison_page_*.html"
    echo "  2. Label each pair and export results"
    echo "  3. Combine exported JSONs into ${ANNOTATIONS}"
    exit 1
fi

echo "Found annotations: $ANNOTATIONS"
echo "Lines: $(wc -l < "$ANNOTATIONS")"

# Compute agreement for each scoring strategy
for strategy in holistic multi_aspect cot; do
    VLM_SCORES="data/exp1_${strategy}/metadata.jsonl"

    if [ ! -f "$VLM_SCORES" ]; then
        echo "WARNING: VLM scores not found for strategy '${strategy}': $VLM_SCORES"
        echo "  Run Exp 1 Phase 1 first: bash scripts/run_exp1.sh"
        continue
    fi

    echo ""
    echo "--- Computing agreement: ${strategy} ---"
    python scripts/prepare_human_eval.py compute \
        --vlm-scores "$VLM_SCORES" \
        --human-annotations "$ANNOTATIONS" \
        --output "${EXP_DIR}/agreement_${strategy}.json" \
        2>&1 | tee "${LOG_DIR}/agreement_${strategy}_${TIMESTAMP}.log"
done

# Summary
echo ""
echo "============================================================"
echo "Agreement Results Summary"
echo "============================================================"

for strategy in holistic multi_aspect cot; do
    result_file="${EXP_DIR}/agreement_${strategy}.json"
    if [ -f "$result_file" ]; then
        echo ""
        echo "--- ${strategy} ---"
        cat "$result_file"
    fi
done

echo ""
echo "=== Experiment 1 complete! ==="
