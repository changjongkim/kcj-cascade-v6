#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v13c_qwen_strong_1_4n
#SBATCH -o benchmark/logs/v13c_qwen_strong_1_4n_%j.out
#SBATCH -e benchmark/logs/v13c_qwen_strong_1_4n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20
export FI_CXI_RETRY_COUNT=100
export FI_CXI_RETRY_WAIT=200
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_OFLOW_BUF_COUNT=7
export FI_CXI_REQ_BUF_SIZE=16777216
export FI_MR_CACHE_MONITOR=disabled
export MPICH_OFI_NUM_NICS=1

SYSTEM="cascade"

echo "--- N=1 ---"
  srun --overlap -N 1 -n 1 --ntasks-per-node=1 \
    python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system $SYSTEM --num-requests 64 --block-size-mb 320 \
    --run-id "${SLURM_JOB_ID}_STRONG_N1"
sleep 5

echo "--- N=2 ---"
  srun --overlap -N 2 -n 2 --ntasks-per-node=1 \
    python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system $SYSTEM --num-requests 64 --block-size-mb 320 \
    --run-id "${SLURM_JOB_ID}_STRONG_N2"
sleep 5

echo "--- N=4 ---"
  srun --overlap -N 4 -n 4 --ntasks-per-node=1 \
    python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system $SYSTEM --num-requests 64 --block-size-mb 320 \
    --run-id "${SLURM_JOB_ID}_STRONG_N4"
sleep 5

echo "Done: v13c_qwen_strong_1_4n"
