#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v16_tail_cas_64n
#SBATCH -o benchmark/logs/v16_tail_cdf_cascade_64n_%j.out
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd ${REPO_ROOT}
source setup_env.sh

SYSTEM="cascade"
BLOCK_SIZES="160 320"
N=64

echo "=== Tail Latency CDF - Cascade Only (64 Nodes) ==="

for bs in $BLOCK_SIZES; do
    model_name="Llama"
    if [ "$bs"-eq "320"]; then model_name="Qwen"; fi
    echo "Testing $SYSTEM with $model_name ($bs MB blocks)"
    srun  -N $N -n $N --ntasks-per-node=1 \
        python3 script_example/02_tail_latency_burst/tail_driver.py \
        --system $SYSTEM --block-size-mb $bs --num-write-blocks 30 --num-read-ops 300
    sleep 5
done

echo "Cascade 64N experiment completed"
