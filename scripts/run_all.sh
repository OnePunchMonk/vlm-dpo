#!/bin/bash
# ============================================================
# VLM-DPO: Run All Experiments
# ============================================================
# Master script that runs experiments in dependency order.
#
# Dependency graph:
#   Exp 1 (standalone)
#   Exp 2 (standalone) ──→ Exp 4 (depends on Exp 2 checkpoint)
#   Exp 3 (standalone) ──→ Exp 4 (comparison baseline)
#                       ──→ Exp 5 (uses Exp 3 data)
#
# Usage:
#   bash scripts/run_all.sh          # Run all experiments
#   bash scripts/run_all.sh 3        # Run only experiment 3
#   bash scripts/run_all.sh 2 4      # Run experiments 2 and 4
# ============================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="outputs/master_log_${TIMESTAMP}.log"

mkdir -p outputs

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$MASTER_LOG"
}

run_exp() {
    local exp_num=$1
    local script="scripts/run_exp${exp_num}.sh"

    if [ ! -f "$script" ]; then
        log "ERROR: Script not found: $script"
        return 1
    fi

    log "========================================="
    log "Starting Experiment ${exp_num}"
    log "========================================="

    if bash "$script" 2>&1 | tee -a "$MASTER_LOG"; then
        log "Experiment ${exp_num} PASSED"
    else
        log "Experiment ${exp_num} FAILED (exit code: $?)"
        return 1
    fi
}

# Determine which experiments to run
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=(1 2 3 4 5)
fi

log "VLM-DPO: Master Experiment Runner"
log "Experiments to run: ${EXPERIMENTS[*]}"
log "Master log: $MASTER_LOG"

FAILED=()

for exp in "${EXPERIMENTS[@]}"; do
    if ! run_exp "$exp"; then
        FAILED+=("$exp")
        # Exp 4 depends on Exp 2, so skip 4 if 2 failed
        if [ "$exp" = "2" ]; then
            log "WARNING: Exp 2 failed, skipping Exp 4 (depends on Exp 2 checkpoint)"
            EXPERIMENTS=("${EXPERIMENTS[@]/4/}")
        fi
    fi
done

log ""
log "========================================="
log "Summary"
log "========================================="

if [ ${#FAILED[@]} -eq 0 ]; then
    log "All experiments completed successfully!"
else
    log "Failed experiments: ${FAILED[*]}"
fi

log "Master log: $MASTER_LOG"
