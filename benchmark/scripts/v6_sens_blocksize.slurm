#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_sens_blocksize
#SBATCH -o benchmark/logs/sens_blocksize_%j.out
#SBATCH -e benchmark/logs/sens_blocksize_%j.err

# Setup Environment
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "###################################################################"
echo " SENSITIVITY ANALYSIS 1: Block Size (Metadata Overhead)"
echo " Fixed: 4 Nodes, Weak Scaling, 60 blocks"
echo " Variable: Qwen-7B (56MB), Qwen-32B (256MB), Qwen-72B (320MB)"
echo "###################################################################"

echo ""
echo "========== Qwen-2.5-7B (56MB blocks) =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-7b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 60

echo ""
echo "========== Qwen-2.5-32B (256MB blocks) =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-32b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 60

echo ""
echo "========== Qwen-2.5-72B (320MB blocks) =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 60
