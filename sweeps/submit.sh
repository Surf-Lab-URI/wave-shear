#!/bin/bash
# Submit every row of a sweep table as one Slurm job array:
#   sweeps/submit.sh sweeps/amplitude-1.csv [extra sbatch options, e.g. -t 12:00:00]
# Each table is one sweep, and its runs go in runs/<sweep>/, named after the table (set RUNS_DIR to use
# another directory). Row <name> runs in runs/<sweep>/<name>/. Rows whose run already finished
# (runs/<sweep>/<name>/done exists) exit straight away, so resubmitting a table only reruns what's missing.
#
# The table's script column picks the simulation for each row: shear-instability-test.jl (the default when
# there's no script column) or shear-instability-wavefield.jl. The other columns are that script's options.
#
# With --plot, make the movies instead, saved as runs/<sweep>/<name>/<name>_3D.mp4 (rows that already
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
runs=$(realpath -m "${RUNS_DIR:-$repo/runs/$(basename "$table" .csv)}")
mkdir -p "$runs"

n=$(( $(grep -cv '^[[:space:]]*$' "$table") - 1 )) # rows, not counting the header

# %x is the job name (ws-sweep or ws-plot)
sbatch --array=1-$n --output="$runs/slurm-%x-%A_%a.out" "$@" "$repo/sweeps/$script" "$table" "$runs" "$repo"
