#!/usr/bin/env bash
#SBATCH --job-name=reviewer-kmers
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=reviewer_benchmark_%A_%a.out
#SBATCH --error=reviewer_benchmark_%A_%a.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch --array=0-N $0 CONFIG CONDITION_MANIFEST SPLIT_MANIFEST" >&2
  exit 2
fi

config_path=$1
condition_manifest=$2
split_manifest=$3
condition_index=${SLURM_ARRAY_TASK_ID:?Submit this script with --array=0-N}

"${PYTHON:-python}" src/reviewer_benchmark.py run-condition \
  --config "$config_path" \
  --condition-manifest "$condition_manifest" \
  --condition-index "$condition_index" \
  --split-manifest "$split_manifest"
