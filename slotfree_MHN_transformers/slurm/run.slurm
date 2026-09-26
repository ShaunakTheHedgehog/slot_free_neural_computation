#!/bin/bash
# Generic array runner: one line of a config file per array task. Each non-comment
# line of the config file is a full set of arguments for main.py (one model run).
#
# Don't call sbatch on this directly -- use the per-experiment launchers, which
# size the array to the config for you:
#   bash slurm/submit_main_L4.sh      # experiment 1 (L=4, 10 models)
#   bash slurm/submit_scaled_L26.sh   # experiment 2 (L=26, 4 models)
#
# Or submit any config file manually from slotfree_MHN_transformers/:
#   CONFIG=slurm/configs/main_L4.txt
#   mkdir -p slurm/logs
#   sbatch --array=1-$(grep -cvE '^\s*(#|$)' $CONFIG) slurm/run.slurm $CONFIG
#
#SBATCH --job-name=mhn-tf
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00               # enough for L=26; override with --time for smaller runs
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=BEGIN,END,FAIL                 # Email notifications
#SBATCH --mail-user=sb0108@princeton.edu

set -eo pipefail
CONFIG=$1

cd "$SLURM_SUBMIT_DIR"

# this task's arguments: line number SLURM_ARRAY_TASK_ID, ignoring comments and blank lines
read -ra ARGS <<< "$(grep -vE '^\s*(#|$)' "$CONFIG" | sed -n "${SLURM_ARRAY_TASK_ID}p")"

# activate the environment -- adapt these two lines to your cluster
module load anaconda3                 # or: source ~/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)" && conda activate slotfree-mhn-tf

# headless plotting, no progress-bar spam in the logs, one thread per CPU
export MPLBACKEND=Agg TQDM_DISABLE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[task $SLURM_ARRAY_TASK_ID] $(hostname)  python main.py ${ARGS[*]}"
python main.py "${ARGS[@]}"
