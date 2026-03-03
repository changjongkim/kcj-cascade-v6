#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J idx_redis
#SBATCH -o benchmark/logs/v14_idx_redis_%j.out
#SBATCH -e benchmark/logs/v14_idx_redis_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="redis"
PORT=16379

echo "=== Index Scalability: ${SYSTEM} ==="

# Startup Redis on submitting node
REDIS_HOST=$(hostname)
mkdir -p benchmark/tmp/hosts_${SLURM_JOB_ID}
echo $REDIS_HOST > benchmark/tmp/hosts_${SLURM_JOB_ID}/redis_host

redis-server --port $PORT --bind 0.0.0.0 --protected-mode no --maxmemory 100gb --maxmemory-policy allkeys-lru --save "" --appendonly no &
REDIS_PID=$!
sleep 10

export REDIS_PORT=$PORT

srun --overlap -N 8 -n 8 --ntasks-per-node=1 \
    python3 benchmark/scripts/v14_index_scalability_bench4.py \
    --system $SYSTEM \
    --test-reqs 128 \
    --block-size 16

kill $REDIS_PID
echo "✅ Done"
