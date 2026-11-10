#!/bin/bash
# Submit every row of a sweep table as one Slurm job array:
#   sweeps/submit.sh sweeps/amplitude-1.csv [extra sbatch options, e.g. -t 12:00:00]
# Row <name> runs in runs/<name>/ (set RUNS_DIR to use another directory). Rows whose run already
# finished (runs/<name>/done exists) exit straight away, so resubmitting a table only reruns what's missing.
#
# With --plot, make the movies instead, saved as runs/<name>/<name>_3D.mp4 (rows that already
# have a movie are skipped):
#   sweeps/submit.sh --plot sweeps/amplitude-1.csv
set -euo pipefail

script=run-sweep.sh
if [[ ${1:-} == --plot ]]; then
    script=plot-sweep.sh
    shift
fi

repo=$(cd "$(dirname "$0")/.." && pwd)
table=$(realpath "$1"); shift
runs=$(realpath -m "${RUNS_DIR:-$repo/runs}")
mkdir -p "$runs"

n=$(( $(grep -cv '^[[:space:]]*$' "$table") - 1 )) # rows, not counting the header

# %x is the job name (ws-sweep or ws-plot)
sbatch --array=1-$n --output="$runs/slurm-%x-%A_%a.out" "$@" "$repo/sweeps/$script" "$table" "$runs" "$repo"
