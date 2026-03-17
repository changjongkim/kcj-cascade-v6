#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v20_cas_1n_fix
#SBATCH -o benchmark/logs/v20_var_debug_cascade_1n_%j.out
#SBATCH -q debug
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "=== 🏁 Variable Block Fixed: Cascade (1 Node Only) ==="

srun -N 1 -n 1 python3 benchmark/scripts/v20_variable_block_bench.py \
    --system cascade \
    --block-size-mb 160 --num-write-blocks 50 --num-read-ops 200 --sigma 0.8

echo "✅ Cascade 1-Node Fix Completed."
