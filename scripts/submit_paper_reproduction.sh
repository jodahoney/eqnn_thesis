#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_ROOT/scripts/sherlock_paper_reproduction_by_trainsize.sbatch}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/slurm/$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "Submitting job with logs under: $LOG_DIR"

sbatch \
  --output="$LOG_DIR/%x_%A_%a.out" \
  --error="$LOG_DIR/%x_%A_%a.err" \
  "$SBATCH_SCRIPT"