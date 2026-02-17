#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_qwen_72b_cold_strong_1n
#SBATCH -o benchmark/logs/qwen_72b_cold_strong_1n_%j.out
#SBATCH -e benchmark/logs/qwen_72b_cold_strong_1n_%j.err

# Setup Environment
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Force specific build directory
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "###################################################################"
echo " COLD START STRONG SCALING (Lustre -> GPU): Qwen 2.5-72B @ 1 Node"
echo " Data: 128 Blocks (41 GB) Fixed (Regular Queue, 2hr)"
echo "###################################################################"

srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode strong \
    --data-type real \
    --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --cold
