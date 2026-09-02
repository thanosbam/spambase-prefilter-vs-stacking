#!/bin/bash
#SBATCH --job-name=spambase_prefilter
#SBATCH --time=02:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=1-70

# Submit from inside "/home3/s6018130/thesis ensemble/proof of concept/":
#   sbatch submit_habrok.sh
#
# Each array task runs one split_seed from SEEDS below, so N seeds finish in
# roughly the time one seed takes (instead of N times that, run sequentially).
# Adjust --array=1-N to match the number of entries in SEEDS.
#
# NOTE: --cpus-per-task, --time, --mem, and the module name below are
# starting guesses, not verified Habrok values. Check `module spider R`
# for the exact R module available to you, and adjust the resource
# requests based on your first run's actual usage (`seff <jobid>` after
# it finishes, or check the sacct output).

set -euo pipefail

module purge
module load R/4.5.1-gfbf-2025a

cd "/home3/s6018130/thesis ensemble/proof of concept"
mkdir -p logs

SEEDS=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000)
CONFIGURATION_ID="SpambasePrefilter5of6OOFThresholds"

SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID - 1))]}"

Rscript prefilter_majority_voting_comparison_spambase_habrok.R "$SEED" "$CONFIGURATION_ID"
