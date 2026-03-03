#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_dedup_pdc_scaled
#SBATCH -o benchmark/logs/v12_dedup_pdc_scaled_%j.out
#SBATCH -e benchmark/logs/v12_dedup_pdc_scaled_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=32768
export FI_CXI_OFLOW_BUF_SIZE=2097152
export FI_CXI_DEFAULT_VNI=0

SYSTEM="pdc"
BLOCK_MB=160
PREFIX_BLOCKS=64

echo "=== Evaluation 1: Global Deduplication & Prefix Sharing (Scaled) ==="
echo "System: ${SYSTEM}"

for i in 16 32; do
    echo ""
    echo "▶ SCALE: N=${i} nodes"
    srun --overlap -N $i -n $i --ntasks-per-node=1 \
        python3 benchmark/scripts/v12_dedup_prefix_bench.py \
        --system $SYSTEM \
        --block-size-mb $BLOCK_MB \
        --prefix-blocks $PREFIX_BLOCKS \
        --run-id "${SLURM_JOB_ID}_DEDUP_N${i}"
    sleep 5
done

echo "✅ Done"
