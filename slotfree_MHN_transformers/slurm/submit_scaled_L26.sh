#!/bin/bash
# Launch experiment 2 (scaled_L26): item_in_mhn MHN models + baseline at L = C = 26.
# Submits one SLURM array task per model line in the config file.
# Usage (from anywhere):  bash slurm/submit_scaled_L26.sh [RUN_TAG]
#
# RUN_TAG (default: gaussinit) names a fresh rerun, so old results/logs are untouched:
#   results -> results_<RUN_TAG>/scaled_L26/...
#   logs    -> slurm/logs_<RUN_TAG>/mhn-L26_<jobid>_<task>.out / .err
set -eo pipefail

RUN_TAG=${1:-gaussinit}

cd "$(dirname "$0")/.."                          # -> slotfree_MHN_transformers/
LOG_DIR=slurm/logs_$RUN_TAG
RESULTS_ROOT=results_$RUN_TAG
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"              # slurm won't create the log folder itself

CONFIG=slurm/configs/scaled_L26.txt
N=$(grep -cvE '^\s*(#|$)' "$CONFIG")             # number of model runs in the file

sbatch --job-name=mhn-L26 --time=12:00:00 --array=1-"$N" \
       --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" \
       slurm/run.slurm "$CONFIG" --results_root "$RESULTS_ROOT"
