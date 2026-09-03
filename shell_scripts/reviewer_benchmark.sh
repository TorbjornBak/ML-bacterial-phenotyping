#!/usr/bin/env bash
#SBATCH --job-name=reviewer-kmers
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=reviewer_benchmark_%A_%a.out
#SBATCH --error=reviewer_benchmark_%A_%a.err

set -euo pipefail

# Resolve the runner from this script, so submitting from a login directory
# (or with an absolute config/manifest path) does not depend on Slurm's
# working-directory policy.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch --array=0-N $0 CONFIG CONDITION_MANIFEST SPLIT_MANIFEST [MODEL] [REUSE_FEATURES]" >&2
  exit 2
fi

config_path=$1
condition_manifest=$2
split_manifest=$3
model=${4:-logistic_regression}
reuse_features=${5:-false}
condition_index=${SLURM_ARRAY_TASK_ID:?Submit this script with --array=0-N}

# Capture argument paths before changing directory.  Relative dataset and
# output paths in the original config are intentionally resolved from the
# repository root; prepared configs contain absolute paths.
case "$config_path" in
  /*) ;;
  *) config_path="$PWD/$config_path" ;;
esac
case "$condition_manifest" in
  /*) ;;
  *) condition_manifest="$PWD/$condition_manifest" ;;
esac
case "$split_manifest" in
  /*) ;;
  *) split_manifest="$PWD/$split_manifest" ;;
esac
cd "$REPO_ROOT"

runner_args=(
  --config "$config_path"
  --condition-manifest "$condition_manifest"
  --condition-index "$condition_index"
  --split-manifest "$split_manifest"
  --model "$model"
)
if [[ "$reuse_features" == "true" ]]; then
  runner_args+=(--reuse-features)
fi

"${PYTHON:-python}" "$REPO_ROOT/src/reviewer_benchmark.py" run-condition \
  "${runner_args[@]}"
