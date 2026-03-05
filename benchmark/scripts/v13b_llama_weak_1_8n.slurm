#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v13b_llama_weak_1_8n
#SBATCH -o benchmark/logs/v13b_llama_weak_1_8n_%j.out
#SBATCH -e benchmark/logs/v13b_llama_weak_1_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# === Aggressive OFI/CXI fixes ===
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

echo "=== v13b: §20 Llama Weak Scaling 1-8N (160MB, 8 req/node) ==="

for N in 1 2 4 8; do
    echo ""
    echo "--- Weak Scaling N=${N} nodes (REQS=$((8 * N))) ---"
    REQS=$((8 * N))

    srun --overlap -N $N -n $N --ntasks-per-node=1 \
        python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id "${SLURM_JOB_ID}_WEAK_N${N}"

    echo "  [cleanup] Waiting 30s for CXI resource release..."
    sleep 5
done

echo "✅ Done: v13b Llama Weak 1-8N"
