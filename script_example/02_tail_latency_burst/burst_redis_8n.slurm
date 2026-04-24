#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v17_bursty_redis_8n
#SBATCH -o benchmark/logs/v17_bursty_redis_8n_%j.out
#SBATCH -e benchmark/logs/v17_bursty_redis_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="redis"
N=8
BLOCK_SIZE=160

echo "=== V17 Bursty Traffic Test: ${SYSTEM} N=${N} ==="

mkdir -p benchmark/tmp/hosts_$SLURM_JOB_ID
hostname > benchmark/tmp/hosts_$SLURM_JOB_ID/redis_host
redis-server --port 16379 --protected-mode no --save "" --appendonly no &
REDIS_PID=$!
sleep 5

srun -N $N -n $N --ntasks-per-node=1 \
    python3 benchmark/scripts/v17_bursty_traffic_bench.py \
    --system $SYSTEM \
    --node-count $N \
    --block-size-mb $BLOCK_SIZE

echo "🧹 Cleaning up Redis and data..."
kill $REDIS_PID
rm -rf benchmark/tmp/hosts_$SLURM_JOB_ID
rm -rf benchmark/tmp/bursty_res_${SLURM_JOB_ID}

echo "✅ Done"
