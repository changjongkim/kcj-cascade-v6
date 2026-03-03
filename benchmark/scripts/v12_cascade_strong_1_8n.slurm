#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_cascade_strong_1_8n
#SBATCH -o benchmark/logs/v12_cascade_strong_1_8n_%j.out
#SBATCH -e benchmark/logs/v12_cascade_strong_1_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Slingshot fixes
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=32768
export FI_CXI_OFLOW_BUF_SIZE=2097152
export FI_CXI_CQ_FILL_PERCENT=10
export FI_CXI_RETRY_COUNT=50
export FI_CXI_RETRY_WAIT=100

SYSTEM="cascade"
JOB_UID=${SLURM_JOB_ID}

echo "=== v12 FIX: CASCADE STRONG SCALING 1-8N (128 req fixed) ==="

for N in 1 2 4 8; do
    echo ""
    echo "--- Strong Scaling N=${N} nodes (REQS=128) ---"
    REQS=128
    RUN_ID="${JOB_UID}_STRONG_N${N}"

    srun --overlap -N $N -n $N --ntasks-per-node=1 \
        python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID

    sleep 10
done

echo "✅ Done: v12 Cascade Strong Scaling 1-8N"
