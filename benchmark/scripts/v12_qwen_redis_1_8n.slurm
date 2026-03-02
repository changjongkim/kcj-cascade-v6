#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_qwen_redis_1_8n
#SBATCH -o benchmark/logs/v12_qwen_redis_1_8n_%j.out
#SBATCH -e benchmark/logs/v12_qwen_redis_1_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=65536
export FI_CXI_OFLOW_BUF_SIZE=4194304
export FI_CXI_CQ_FILL_PERCENT=10

# ─── Start Redis on ALL 8 nodes (one per node) ───
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
BLOCK_MB=320

echo "=== Qwen-2.5-72B (320MB synthetic blocks) | ${SYSTEM} | Weak+Strong 1-8N ==="

# ── Weak Scaling: 8 req/node ──
echo ""
echo "▶ WEAK SCALING"
for N in 1 2 4 8; do
    echo "  N=${N} nodes (REQS=$((8 * N)))"
    srun --overlap -N $N -n $N --ntasks-per-node=1 \
        python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests $((8 * N)) \
        --block-size-mb $BLOCK_MB \
        --run-id "${JOB_UID}_QWEN_WEAK_N${N}"
    sleep 10
done

# ── Strong Scaling: 128 req fixed ──
echo ""
echo "▶ STRONG SCALING (128 req fixed)"
for N in 1 2 4 8; do
    echo "  N=${N} nodes (REQS=128)"
    srun --overlap -N $N -n $N --ntasks-per-node=1 \
        python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests 128 \
        --block-size-mb $BLOCK_MB \
        --run-id "${JOB_UID}_QWEN_STRONG_N${N}"
    sleep 10
done

# Cleanup
echo "Stopping Redis on all nodes..."
srun --overlap -N 8 -n 8 --ntasks-per-node=1 pkill -u $USER redis-server || true

echo ""
echo "✅ Done: v12 Qwen 320MB Weak+Strong 1-8N | ${SYSTEM}"
