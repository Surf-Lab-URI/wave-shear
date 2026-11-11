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
# With --with-plots, also make each run's movie, saved as runs/<sweep>/<name>/<name>_3D.mp4, as soon as that
# run finishes successfully. The movie of a run that fails is cancelled. Extra sbatch options only apply to the
# simulations:
#   sweeps/submit.sh --with-plots sweeps/amplitude-1.csv
#
# With --plot, only make the movies, for runs that have finished (rows that already have a movie are skipped):
#   sweeps/submit.sh --plot sweeps/amplitude-1.csv
set -euo pipefail

mode=run
case ${1:-} in
    --plot)       mode=plot; shift ;;
    --with-plots) mode=run-and-plot; shift ;;
esac

repo=$(cd "$(dirname "$0")/.." && pwd)
table=$(realpath "$1"); shift
runs=$(realpath -m "${RUNS_DIR:-$repo/runs/$(basename "$table" .csv)}")
mkdir -p "$runs"

n=$(( $(grep -cv '^[[:space:]]*$' "$table") - 1 )) # rows, not counting the header

# %x is the job name (ws-sweep or ws-plot)
array=(--array=1-$n --output="$runs/slurm-%x-%A_%a.out")
job_args=("$table" "$runs" "$repo")

if [[ $mode == plot ]]; then
    sbatch "${array[@]}" "$@" "$repo/sweeps/plot-sweep.sh" "${job_args[@]}"
    exit
fi

run_job=$(sbatch --parsable "${array[@]}" "$@" "$repo/sweeps/run-sweep.sh" "${job_args[@]}")
run_job=${run_job%%;*} # --parsable can append ";cluster"
echo "Submitted the simulations as job $run_job"

if [[ $mode == run-and-plot ]]; then
    # aftercorr: movie task i starts when simulation task i finishes successfully
    plot_job=$(sbatch --parsable "${array[@]}" --dependency=aftercorr:$run_job --kill-on-invalid-dep=yes \
                      "$repo/sweeps/plot-sweep.sh" "${job_args[@]}")
    echo "Submitted the movies as job ${plot_job%%;*}: each one starts as soon as its run finishes successfully"
fi
