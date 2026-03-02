#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_redis_weak_1_8n
#SBATCH -o benchmark/logs/v12_redis_weak_1_8n_%j.out
#SBATCH -e benchmark/logs/v12_redis_weak_1_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Slingshot fixes
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=65536

# ─── Start Redis on ALL 8 nodes (one per node) ───
# FIX: each rank must have its OWN redis-server running on its node.
# Workers write to local redis, then read phase reconnects to target_rank's hostname.
echo "Starting Redis on all 8 nodes (one per node)..."
srun --overlap -N 8 -n 8 --ntasks-per-node=1 \
    redis-server --port 16379 --bind 0.0.0.0 \
    --protected-mode no --save "" \
    --maxmemory 200gb --maxmemory-policy allkeys-lru &
REDIS_PID=$!
sleep 20
echo "Redis started (background PID=$REDIS_PID)"

SYSTEM="lmcache-redis"
JOB_UID=${SLURM_JOB_ID}

echo "=== WEAK SCALING: 8 req/node ==="

for N in 1 2 4 8; do
    echo ""
    echo "--- Weak Scaling N=${N} nodes (REQS=$((8 * N))) ---"
    REQS=$((8 * N))
    RUN_ID="${JOB_UID}_WEAK_N${N}"

    srun --overlap -N $N -n $N --ntasks-per-node=1 \
        python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID

    sleep 10
done

# Cleanup
echo "Stopping Redis on all nodes..."
srun --overlap -N 8 -n 8 --ntasks-per-node=1 pkill -u $USER redis-server || true
wait $REDIS_PID 2>/dev/null || true
echo "✅ Done: v12 Redis Weak Scaling 1-8N"
