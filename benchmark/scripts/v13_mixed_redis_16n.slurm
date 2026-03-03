#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J mixed_16n_redis
#SBATCH -o benchmark/logs/mixed_16n_redis_%j.out
#SBATCH -e benchmark/logs/mixed_16n_redis_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="redis"
DURATION=60
BLOCK_MB=16
PORT=16379

echo "=== Concurrent Mixed R/W (16-Node): ${SYSTEM} ==="

# Startup Redis on submitting node
REDIS_HOST=$(hostname)
mkdir -p benchmark/tmp/hosts_${SLURM_JOB_ID}
echo $REDIS_HOST > benchmark/tmp/hosts_${SLURM_JOB_ID}/redis_host

redis-server --port $PORT --bind 0.0.0.0 --protected-mode no --maxmemory 100gb --maxmemory-policy allkeys-lru --save "" --appendonly no &
REDIS_PID=$!
sleep 10

export REDIS_PORT=$PORT

srun --overlap -N 16 -n 16 --ntasks-per-node=1 \
    python3 benchmark/scripts/v13_mixed_workload_bench.py \
    --system $SYSTEM \
    --duration $DURATION \
    --block-size-mb $BLOCK_MB \
    --prefill-blocks 200

kill $REDIS_PID
echo "✅ Done"
