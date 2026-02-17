#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J qwen_72b_comp
#SBATCH -o benchmark/logs/qwen_72b_comp_%N_%j.out
#SBATCH -e benchmark/logs/qwen_72b_comp_%N_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

PYTHON_BIN=$(which python3)
SCRIPT=benchmark/scripts/v6_contention_scaling_all.py

# 모델: qwen-2.5-72b
# 데이터: real (현실적인 스트레스 테스트)
# 모드: weak (노드당 부하 고정)
# 시스템: 전수 비교
SYSTEMS="Cascade,HDF5,vLLM-GPU,PDC,LMCache"

echo "###################################################################"
echo " FULL COMPARISON: Qwen 2.5-72B @ $SLURM_NNODES Nodes"
echo "###################################################################"

srun $PYTHON_BIN $SCRIPT \
    --mode weak \
    --data-type real \
    --model qwen-2.5-72b \
    --systems $SYSTEMS

echo "Benchmark Completed for $SLURM_NNODES Nodes."
