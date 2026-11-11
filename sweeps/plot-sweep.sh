#!/bin/bash
#SBATCH --job-name="ws-plot"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH -t 1-00:00:00

# Makes the 3D movie for one row of a sweep table, chosen by the array index, as runs/<sweep>/<name>/<name>_3D.mp4.
# Submit with sweeps/submit.sh --plot rather than directly.
set -euo pipefail

table=$1
runs=$2
repo=$3

export PATH=$HOME/.juliaup/bin:$PATH

name=$(grep -v '^[[:space:]]*$' "$table" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" | cut -d, -f1)
rundir=$runs/$name
movie=${name}_3D.mp4

if [[ ! -e $rundir/done ]]; then
    echo "$name hasn't finished running, so there's nothing to plot"
    exit 1
fi

if [[ -e $rundir/$movie ]]; then
    echo "$rundir/$movie already exists, skipping"
    exit 0
fi

cd "$rundir"
echo "Plotting $name, job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"

# Write to a temporary name first, so a job that's cut short doesn't leave a movie that looks finished
julia "$repo/plotting/shear-instability-test-plotting.jl" "${name}_slices.jld2" "${name}_3D.tmp.mp4" 2>&1 | tee plot.log
mv "${name}_3D.tmp.mp4" "$movie"
