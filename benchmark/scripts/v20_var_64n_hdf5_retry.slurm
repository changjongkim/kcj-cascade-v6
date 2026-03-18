#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v20_var_64n_hdf5
#SBATCH -o benchmark/logs/v20_var_64n_hdf5_%j.out
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

SYSTEM="hdf5"

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "=== 🏁 Variable Block 64-Node Scale: ${SYSTEM} (Retry) ==="

# Run the variable block benchmark
srun -N 64 -n 64 python3 benchmark/scripts/v20_variable_block_bench.py \
    --system ${SYSTEM} \
    --block-size-mb 160 \
    --num-write-blocks 30 \
    --num-read-ops 100 \
    --sigma 0.8

echo "✅ ${SYSTEM} 64-Node Variable Block experiment completed."
