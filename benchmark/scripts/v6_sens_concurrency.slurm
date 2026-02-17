#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_sens_concurrency
#SBATCH -o benchmark/logs/sens_concurrency_%j.out
#SBATCH -e benchmark/logs/sens_concurrency_%j.err

# Setup Environment
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "###################################################################"
echo " SENSITIVITY ANALYSIS 3: Concurrent Request Scaling"
echo " Fixed: 4 Nodes, Weak Scaling, Qwen-72B"
echo " Variable: Block Count 20, 60, 120, 200"
echo "###################################################################"

echo ""
echo "========== 20 Blocks (6.4 GB total) =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 20

echo ""
echo "========== 60 Blocks (19.2 GB total) =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 60

echo ""
echo "========== 120 Blocks (38.4 GB total) =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 120

echo ""
echo "========== 200 Blocks (64 GB total) =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 200
