#!/bin/bash
# Submit every row of a sweep table as one Slurm job array:
#   sweeps/submit.sh sweeps/amplitude-1.csv [extra sbatch options, e.g. -t 12:00:00]
# Row <name> runs in runs/<name>/ (set RUNS_DIR to use another directory). Rows whose run already
# finished (runs/<name>/done exists) exit straight away, so resubmitting a table only reruns what's missing.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
table=$(realpath "$1"); shift
runs=$(realpath -m "${RUNS_DIR:-$repo/runs}")
mkdir -p "$runs"

n=$(( $(grep -cv '^[[:space:]]*$' "$table") - 1 )) # rows, not counting the header

sbatch --array=1-$n --output="$runs/slurm-%A_%a.out" "$@" "$repo/sweeps/run-sweep.sh" "$table" "$runs" "$repo"
