#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J c2h_cascade
#SBATCH -o benchmark/logs/v14_c2h_cascade_%j.out
#SBATCH -e benchmark/logs/v14_c2h_cascade_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="cascade"
echo "=== Cold to Hot Promotion: ${SYSTEM} ==="

# Needed permissions/modules for vmtouch (or using it silently)
module load vmtouch 2>/dev/null || true

srun --overlap -N 8 -n 8 --ntasks-per-node=1 \
    python3 benchmark/scripts/v14_cold_to_hot_cascade.py \
    --system $SYSTEM \
    --num-blocks 1000 \
    --block-size 16

echo "✅ Done"
