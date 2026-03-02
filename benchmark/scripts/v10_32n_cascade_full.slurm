#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v10_32n_cascade_full
#SBATCH -o benchmark/logs/v10_32n_cascade_full_%j.out
#SBATCH -e benchmark/logs/v10_32n_cascade_full_%j.err

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

NODES=(1 2 4 8 16 32)
SYSTEM="cascade"

echo "--- 📈 CASCADE WEAK SCALING (8 Req/Node) ---"
for N in "${NODES[@]}"; do
    echo "   [Scale: $N Nodes]"
    REQS=$((8 * N))
    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM --num-requests $REQS --run-id "${SLURM_JOB_ID}_WEAK_N${N}"
    sleep 5
done

echo "--- 📉 CASCADE STRONG SCALING (128 Req Total) ---"
for N in "${NODES[@]}"; do
    echo "   [Scale: $N Nodes]"
    REQS=128
    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM --num-requests $REQS --run-id "${SLURM_JOB_ID}_STRONG_N${N}"
    sleep 5
done

echo "✅ Finished Cascade Full Scaling"
