#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_qwen_72b_cold_strong
#SBATCH -o benchmark/logs/qwen_72b_cold_strong_%j.out
#SBATCH -e benchmark/logs/qwen_72b_cold_strong_%j.err

# Setup Environment
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Force specific build directory
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "###################################################################"
echo " COLD START STRONG SCALING (Lustre -> GPU): Qwen 2.5-72B"
echo " Data: 128 Blocks (41 GB) Fixed across cluster"
echo "###################################################################"

srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode strong \
    --data-type real \
    --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --cold
