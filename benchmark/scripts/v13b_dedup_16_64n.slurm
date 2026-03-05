#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v13b_dedup_16_64n
#SBATCH -o benchmark/logs/v13b_dedup_16_64n_%j.out
#SBATCH -e benchmark/logs/v13b_dedup_16_64n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# === Aggressive OFI/CXI fixes for large-scale MPI ===
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
BLOCK_MB=160
PREFIX_BLOCKS=64

echo "=== v13b: Prefix Sharing 16-64N (batched broadcast, OFI fix) ==="
echo "System: ${SYSTEM}, ${PREFIX_BLOCKS} blocks x ${BLOCK_MB}MB"
echo ""

for i in 16 32 64; do
    echo ""
    echo "▶ SCALE: N=${i} nodes"
    srun --overlap -N $i -n $i --ntasks-per-node=1 \
        python3 benchmark/scripts/v12_dedup_prefix_bench.py \
        --system $SYSTEM \
        --block-size-mb $BLOCK_MB \
        --prefix-blocks $PREFIX_BLOCKS \
        --run-id "${SLURM_JOB_ID}_DEDUP_N${i}"

    echo "  [cleanup] Waiting 30s for CXI resource release..."
    sleep 5
done

echo "✅ Done: v13b Dedup 16-64N"
