#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_cascade_strong_16n
#SBATCH -o benchmark/logs/v12_cascade_strong_16n_%j.out
#SBATCH -e benchmark/logs/v12_cascade_strong_16n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Slingshot Fixes
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=65536
export FI_CXI_OFLOW_BUF_SIZE=4194304
export FI_CXI_CQ_FILL_PERCENT=10
export FI_CXI_RETRY_COUNT=50
export FI_CXI_RETRY_WAIT=100

# FIX: sync_metadata() in read loop was causing O(N^2) Allgather at 16N+
# This script re-runs Cascade strong scaling (128 req fixed) with the fix applied.

SYSTEM="cascade"
REQS=128

echo "--- v12 FIX: CASCADE STRONG SCALING 16N (${REQS} total reqs) ---"
echo "Bug fix: removed sync_metadata() from per-block read retry loop"
srun --overlap -N 16 -n 16 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system $SYSTEM \
    --num-requests $REQS \
    --run-id "${SLURM_JOB_ID}_STRONG_N16"

echo "✅ Finished v12 Cascade Strong Scaling 16N"
