#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J idx_c2h_disk_8n
#SBATCH -o benchmark/logs/v14_idx_c2h_disk_8n_%j.out
#SBATCH -e benchmark/logs/v14_idx_c2h_disk_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="cascade"
echo "=== Index C2H Scalability (DISK MODE): ${SYSTEM} — 8 Nodes ==="

srun --overlap -N 8 -n 8 --ntasks-per-node=1 \
    python3 benchmark/scripts/v14_index_c2h_bench.py \
    --system $SYSTEM \
    --disk-mode \
    --test-reqs 128 \
    --block-size 16 --scale-steps 50000

echo "✅ Done"
