#!/bin/bash
#SBATCH --job-name="ws-sweep"
#SBATCH --partition=gpu-highvram
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gpus=1
#SBATCH --mem=96000M
#SBATCH --constraint="intel&vram80"
#SBATCH --no-requeue
#SBATCH -t 2-00:00:00

# Runs one row of a sweep table, chosen by the array index. Submit with sweeps/submit.sh rather than directly.
# Each column of the table (other than name) becomes a command-line option, e.g. stop_time -> --stop-time.
set -euo pipefail

table=$1
runs=$2
repo=$3

export PATH=$HOME/.juliaup/bin:$PATH

IFS=, read -ra keys <<< "$(head -n 1 "$table")"
IFS=, read -ra values <<< "$(grep -v '^[[:space:]]*$' "$table" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"

name=${values[0]}
rundir=$runs/$name

if [[ -e $rundir/done ]]; then
    echo "$name already finished, skipping"
    exit 0
fi

mkdir -p "$rundir"
cd "$rundir"

options=()
for i in "${!keys[@]}"; do
    [[ ${keys[i]} == name ]] && continue
    options+=("--${keys[i]//_/-}" "${values[i]}")
done

echo "Run $name, job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi -L

julia "$repo/shear-instability-test.jl" "${options[@]}" --slices-only --output "$name" 2>&1 | tee simulation.log

touch done
