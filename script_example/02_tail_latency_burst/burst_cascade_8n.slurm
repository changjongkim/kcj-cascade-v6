#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v17_bursty_cas_8n
#SBATCH -o benchmark/logs/v17_bursty_cascade_8n_%j.out
#SBATCH -e benchmark/logs/v17_bursty_cascade_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="cascade"
N=8
BLOCK_SIZE=160

echo "=== V17 Bursty Traffic Test: ${SYSTEM} N=${N} ==="

srun  -N $N -n $N --ntasks-per-node=1 \
    python3 script_example/02_tail_latency_burst/burst_driver.py \
    --system $SYSTEM \
    --node-count $N \
    --block-size-mb $BLOCK_SIZE

echo "🧹 Cleaning up data..."
rm -rf benchmark/cascade_bursty_${SLURM_JOB_ID}
rm -rf benchmark/tmp/bursty_res_${SLURM_JOB_ID}

echo "✅ Done"
