#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_dedup_pdc
#SBATCH -o benchmark/logs/v12_dedup_pdc_%j.out
#SBATCH -e benchmark/logs/v12_dedup_pdc_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="pdc"
BLOCK_MB=160
PREFIX_BLOCKS=64

echo "=== Evaluation 1: Global Deduplication & Prefix Sharing ==="
echo "System: ${SYSTEM}"

for i in 1 2 4 8; do
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
