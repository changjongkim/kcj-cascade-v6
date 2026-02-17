#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:20:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_qwen_72b_hot_30hit_16n
#SBATCH -o benchmark/logs/qwen_72b_hot_30hit_16n_%j.out
#SBATCH -e benchmark/logs/qwen_72b_hot_30hit_16n_%j.err

# Setup Environment
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Force specific build directory
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "###################################################################"
echo " HOT START COMPARISON (30% Cache Hit): Qwen 2.5-72B @ 16 Nodes"
echo " 30% Hit (HBM/DRAM) / 70% Miss (Lustre)"
echo " Queue: Regular (20m)"
echo "###################################################################"

srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak \
    --data-type real \
    --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --hit-rate 0.3
