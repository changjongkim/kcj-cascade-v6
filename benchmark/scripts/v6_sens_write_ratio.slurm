#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_sens_write_ratio
#SBATCH -o benchmark/logs/sens_write_ratio_%j.out
#SBATCH -e benchmark/logs/sens_write_ratio_%j.err

# Setup Environment
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "###################################################################"
echo " SENSITIVITY ANALYSIS 2: Put/Get Mixed Workload"
echo " Fixed: 4 Nodes, Weak Scaling, Qwen-72B, 60 blocks"
echo " Variable: Write Ratio 0%, 20%, 50%, 80%"
echo "###################################################################"

echo ""
echo "========== Write Ratio: 0% (Pure Read) =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 60 --write-ratio 0.0

echo ""
echo "========== Write Ratio: 20% =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 60 --write-ratio 0.2

echo ""
echo "========== Write Ratio: 50% =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 60 --write-ratio 0.5

echo ""
echo "========== Write Ratio: 80% =========="
srun python3 benchmark/scripts/v6_contention_scaling_all.py \
    --mode weak --data-type real --model qwen-2.5-72b \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \
    --num-blocks 60 --write-ratio 0.8
