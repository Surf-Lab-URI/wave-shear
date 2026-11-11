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
# The script column picks the simulation (default shear-instability-test.jl). Every other column except name
# becomes a command-line option of that script, e.g. stop_time -> --stop-time.
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

script=shear-instability-test.jl
options=()
for i in "${!keys[@]}"; do
    case ${keys[i]} in
        name) ;;
        script) script=${values[i]} ;;
        *) options+=("--${keys[i]//_/-}" "${values[i]}") ;;
    esac
done

echo "Run $name with $script, job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi -L

julia "$repo/$script" "${options[@]}" --slices-only --output "$name" 2>&1 | tee simulation.log

touch done
