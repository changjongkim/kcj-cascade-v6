#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J qwen_72b_cold
#SBATCH -o benchmark/logs/qwen_72b_cold_%N_%j.out
#SBATCH -e benchmark/logs/qwen_72b_cold_%N_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

PYTHON_BIN=$(which python3)
SCRIPT=benchmark/scripts/v6_contention_scaling_all.py

# 모델: qwen-2.5-72b (Cold Start 실험)
# 모든 시스템이 Lustre 디스크에서 읽기 시작하도록 강제함
SYSTEMS="Cascade,HDF5,vLLM-GPU,PDC,LMCache"

echo "###################################################################"
echo " COLD START COMPARISON (Lustre -> GPU): Qwen 2.5-72B @ $SLURM_NNODES Nodes"
echo "###################################################################"

srun $PYTHON_BIN $SCRIPT \
    --mode weak \
    --data-type real \
    --model qwen-2.5-72b \
    --systems $SYSTEMS \
    --cold

echo "Cold Start Benchmark Completed for $SLURM_NNODES Nodes."
