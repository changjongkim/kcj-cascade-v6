#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J idx_cascade
#SBATCH -o benchmark/logs/v14_idx_cascade_%j.out
#SBATCH -e benchmark/logs/v14_idx_cascade_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="cascade"
echo "=== Index Scalability: ${SYSTEM} ==="

srun --overlap -N 8 -n 8 --ntasks-per-node=1 \
    python3 benchmark/scripts/v14_index_scalability_bench4.py \
    --system $SYSTEM \
    --test-reqs 128 \
    --block-size 16

echo "✅ Done"
