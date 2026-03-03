#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:15:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J dedup_cas_8n
#SBATCH -o benchmark/logs/v12_dedup_cascade_8n_%j.out
#SBATCH -e benchmark/logs/v12_dedup_cascade_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
# Ultra-conservative settings for 8 nodes to avoid Segfault
export FI_CXI_DEFAULT_CQ_SIZE=16384
export FI_CXI_OFLOW_BUF_SIZE=1048576
export FI_CXI_DEFAULT_VNI=0

SYSTEM="cascade"
BLOCK_MB=160
PREFIX_BLOCKS=64

echo "▶ SCALE: N=8 nodes ONLY"
srun --overlap -N 8 -n 8 --ntasks-per-node=1 \
    python3 benchmark/scripts/v12_dedup_prefix_bench.py \
    --system $SYSTEM \
    --block-size-mb $BLOCK_MB \
    --prefix-blocks $PREFIX_BLOCKS \
    --run-id "${SLURM_JOB_ID}_DEDUP_N8"

echo "✅ Done"
