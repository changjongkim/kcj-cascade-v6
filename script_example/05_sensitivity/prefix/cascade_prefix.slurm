#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v13_cascade_weak_prefixrep
#SBATCH -o benchmark/logs/v13_cascade_weak_prefixrep_%j.out
#SBATCH -e benchmark/logs/v13_cascade_weak_prefixrep_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=65536
export FI_CXI_OFLOW_BUF_SIZE=4194304
export FI_CXI_CQ_FILL_PERCENT=10
export FI_CXI_RETRY_COUNT=50
export FI_CXI_RETRY_WAIT=100

SYSTEM="cascade"
JOB_UID=${SLURM_JOB_ID}

echo "=== v13: CASCADE WEAK SCALING 1-8N (Llama 160MB, 8 req/node) ==="
echo "Feature: prefix_replication=ON (broadcast_prefix_data)"
echo ""

for N in 1 2 4 8; do
    echo ""
    echo "--- Weak Scaling N=${N} nodes (REQS=$((8 * N))) ---"
    REQS=$((8 * N))
    RUN_ID="${JOB_UID}_WEAK_N${N}"

    srun --overlap -N $N -n $N --ntasks-per-node=1 \
        python3 script_example/05_sensitivity/prefix/throughput_driver.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID

    sleep 10
done

echo "✅ Done: v13 Cascade Weak Scaling 1-8N (prefix_replication)"
