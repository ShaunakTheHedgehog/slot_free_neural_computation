#!/bin/bash
# Launch experiment 2 (scaled_L26): item_in_mhn MHN models + baseline at L = C = 26.
# Submits one SLURM array task per model line in the config file.
# Usage (from anywhere):  bash slurm/submit_scaled_L26.sh
set -eo pipefail

cd "$(dirname "$0")/.."                           # -> slotfree_MHN_transformers/
mkdir -p slurm/logs

CONFIG=slurm/configs/scaled_L26.txt
N=$(grep -cvE '^\s*(#|$)' "$CONFIG")              # number of model runs in the file

sbatch --job-name=mhn-L26 --time=12:00:00 --array=1-"$N" slurm/run.slurm "$CONFIG"
