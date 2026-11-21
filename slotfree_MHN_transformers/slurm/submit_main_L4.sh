#!/bin/bash
# Launch experiment 1 (main_L4): all 10 model classes at L = C = 4.
# Submits one SLURM array task per model line in the config file.
# Usage (from anywhere):  bash slurm/submit_main_L4.sh [RUN_TAG]
#
# RUN_TAG (default: gaussinit) names a fresh rerun, so old results/logs are untouched:
#   results -> results_<RUN_TAG>/main_L4/...
#   logs    -> slurm/logs_<RUN_TAG>/mhn-L4_<jobid>_<task>.out / .err
set -eo pipefail

RUN_TAG=${1:-gaussinit}

cd "$(dirname "$0")/.."                          # -> slotfree_MHN_transformers/
LOG_DIR=slurm/logs_$RUN_TAG
RESULTS_ROOT=results_$RUN_TAG
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"              # slurm won't create the log folder itself

CONFIG=slurm/configs/main_L4.txt
N=$(grep -cvE '^\s*(#|$)' "$CONFIG")             # number of model runs in the file

sbatch --job-name=mhn-L4 --time=02:00:00 --array=1-"$N" \
       --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" \
       slurm/run.slurm "$CONFIG" --results_root "$RESULTS_ROOT"
