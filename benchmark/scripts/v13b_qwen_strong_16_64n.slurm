#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v13b_qwen_strong_16_64n
#SBATCH -o benchmark/logs/v13b_qwen_strong_16_64n_%j.out
#SBATCH -e benchmark/logs/v13b_qwen_strong_16_64n_%j.err

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
TOTAL_REQS=64

echo "=== v13b: §21 Qwen-72B Strong Scaling 16-64N (320MB, fixed 64 reqs) ==="

for N in 16 32 64; do
    echo ""
    echo "--- Strong Scaling N=${N} nodes (REQS=${TOTAL_REQS} fixed) ---"

    srun --overlap -N $N -n $N --ntasks-per-node=1 \
        python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests $TOTAL_REQS \
        --block-size-mb 320 \
        --run-id "${SLURM_JOB_ID}_QWEN_STRONG_N${N}"

    echo "  [cleanup] Waiting 30s..."
    sleep 5
done

echo "✅ Done: v13b Qwen Strong 16-64N"
