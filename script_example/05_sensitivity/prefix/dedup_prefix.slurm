#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v13_dedup_cascade_prefixrep
#SBATCH -o benchmark/logs/v13_dedup_cascade_prefixrep_%j.out
#SBATCH -e benchmark/logs/v13_dedup_cascade_prefixrep_%j.err

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
BLOCK_MB=160
PREFIX_BLOCKS=64

echo "=== v13: Global Prefix Sharing (prefix_replication=ON) ==="
echo "System: ${SYSTEM}, ${PREFIX_BLOCKS} blocks x ${BLOCK_MB}MB"
echo "Feature: broadcast_prefix_data() — all nodes get local copies"
echo ""

for i in 1 2 4 8; do
    echo ""
    echo "▶ SCALE: N=${i} nodes"
    srun --overlap -N $i -n $i --ntasks-per-node=1 \
        python3 script_example/05_sensitivity/prefix/dedup_prefix_driver.py \
        --system $SYSTEM \
        --block-size-mb $BLOCK_MB \
        --prefix-blocks $PREFIX_BLOCKS \
        --run-id "${SLURM_JOB_ID}_DEDUP_N${i}"
    sleep 5
done

echo "✅ Done: v13 Dedup Prefix Sharing (prefix_replication)"
