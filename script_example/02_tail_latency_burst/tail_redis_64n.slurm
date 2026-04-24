#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v16_tail_redis_64n
#SBATCH -o benchmark/logs/v16_tail_cdf_redis_64n_%j.out
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEMS="redis"
BLOCK_SIZES="160 320"

echo "=== Tail Latency CDF Data Collection - Redis Only (64 Nodes) ==="

for sys in $SYSTEMS; do
    echo ">>> Starting Redis Server on rank 0..."
    mkdir -p benchmark/tmp/hosts_$SLURM_JOB_ID

    if [ $SLURM_PROCID -eq 0 ]; then
        hostname > benchmark/tmp/hosts_$SLURM_JOB_ID/redis_host
        redis-server --port 16379 --protected-mode no --save "" --appendonly no \
            --maxmemory 200gb --maxmemory-policy allkeys-lru &
        REDIS_PID=$!
        echo "Redis started with PID $REDIS_PID"
    fi

    sleep 15

    for bs in $BLOCK_SIZES; do
        model_name="Llama"
        if [ "$bs" -eq "320" ]; then model_name="Qwen"; fi
        echo "▶ Testing $sys with $model_name ($bs MB blocks)"
        srun -N 64 -n 64 python3 benchmark/scripts/v12_tail_latency_bench.py \
            --system $sys --block-size-mb $bs --num-write-blocks 30 --num-read-ops 300
        sleep 10
    done

    if [ $SLURM_PROCID -eq 0 ]; then
        kill $REDIS_PID 2>/dev/null || true
    fi
    rm -rf benchmark/tmp/hosts_$SLURM_JOB_ID
done

echo "✅ Redis-only experiment completed"
