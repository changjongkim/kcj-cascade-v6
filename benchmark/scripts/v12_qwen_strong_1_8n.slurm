#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_qwen_strong_1_8n
#SBATCH -o benchmark/logs/v12_qwen_strong_1_8n_%j.out
#SBATCH -e benchmark/logs/v12_qwen_strong_1_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=65536
export FI_CXI_OFLOW_BUF_SIZE=4194304
export FI_CXI_CQ_FILL_PERCENT=10

JOB_UID=${SLURM_JOB_ID}
BLOCK_MB=320
REQS=128
# Qwen-2.5-72B: 320MB/block (streaming synthetic)
# Strong scaling: 128 req fixed

echo "=== Qwen-2.5-72B (320MB blocks) STRONG SCALING 1-8N (128 req fixed) ==="

SYSTEMS=("cascade" "HDF5-Indep" "lmcache" "pdc" "llm-gpu")

for SYSTEM in "${SYSTEMS[@]}"; do
    echo ""
    echo "━━━ SYSTEM: ${SYSTEM} ━━━"

    for N in 1 2 4 8; do
        echo "  --- Strong N=${N} (REQS=${REQS} fixed, ${BLOCK_MB}MB/block) ---"
        RUN_ID="${JOB_UID}_QWEN_STRONG_${SYSTEM}_N${N}"

        srun --overlap -N $N -n $N --ntasks-per-node=1 \
            python3 benchmark/scripts/v9_serv_metrics_standalone.py \
            --system $SYSTEM \
            --num-requests $REQS \
            --block-size-mb $BLOCK_MB \
            --run-id $RUN_ID

        sleep 10
    done
done

echo ""
echo "✅ Done: v12 Qwen 320MB Strong Scaling 1-8N (all systems)"
