#!/bin/bash
# Launch experiment 1 (main_L4): all 10 model classes at L = C = 4.
# Submits one SLURM array task per model line in the config file.
# Usage (from anywhere):  bash slurm/submit_main_L4.sh
set -eo pipefail

cd "$(dirname "$0")/.."                          # -> slotfree_MHN_transformers/
mkdir -p slurm/logs

CONFIG=slurm/configs/main_L4.txt
N=$(grep -cvE '^\s*(#|$)' "$CONFIG")             # number of model runs in the file

sbatch --job-name=mhn-L4 --time=02:00:00 --array=1-"$N" slurm/run.slurm "$CONFIG"
