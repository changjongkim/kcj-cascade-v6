#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J qwen_7b_comp
#SBATCH -o benchmark/logs/qwen_7b_comp_%N_%j.out
#SBATCH -e benchmark/logs/qwen_7b_comp_%N_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

PYTHON_BIN=$(which python3)
SCRIPT=benchmark/scripts/v6_contention_scaling_all.py

# 모델: qwen-2.5-7b (56MB 블록)
SYSTEMS="Cascade,HDF5,vLLM-GPU,PDC,LMCache"

echo "###################################################################"
echo " SMALL BLOCK COMPARISON: Qwen 2.5-7B @ $SLURM_NNODES Nodes"
echo "###################################################################"

srun $PYTHON_BIN $SCRIPT \
    --mode weak \
    --data-type real \
    --model qwen-2.5-7b \
    --systems $SYSTEMS

echo "Benchmark Completed for $SLURM_NNODES Nodes."
