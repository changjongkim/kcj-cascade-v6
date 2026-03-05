#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J idx_cascade_1n
#SBATCH -o benchmark/logs/v14_idx_cascade__1n_%j.out
#SBATCH -e benchmark/logs/v14_idx_cascade__1n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="cascade"
echo "=== Index Scalability: ${SYSTEM} ==="

srun --overlap -N 1 -n 1 --ntasks-per-node=1 \
    python3 benchmark/scripts/v14_index_scalability_bench4.py \
    --system $SYSTEM \
    --test-reqs 128 \
    --block-size 16 --scale-steps 50000

echo "✅ Done"
