#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J mixed_pdc
#SBATCH -o benchmark/logs/mixed_pdc_%j.out
#SBATCH -e benchmark/logs/mixed_pdc_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="pdc"
DURATION=60
BLOCK_MB=16

echo "=== Concurrent Mixed R/W: ${SYSTEM} ==="

srun --overlap -N 8 -n 8 --ntasks-per-node=1 \
    python3 benchmark/scripts/v13_mixed_workload_bench.py \
    --system $SYSTEM \
    --duration $DURATION \
    --block-size-mb $BLOCK_MB \
    --prefill-blocks 200

echo "✅ Done"
