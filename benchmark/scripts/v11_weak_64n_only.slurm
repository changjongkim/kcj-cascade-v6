#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v11_weak_64n_only
#SBATCH -o benchmark/logs/v11_weak_64n_only_%j.out
#SBATCH -e benchmark/logs/v11_weak_64n_only_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# --- 🚀 Slingshot NIC Stability Fixes ---
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=32768
export FI_CXI_OFLOW_BUF_SIZE=4194304
export FI_CXI_CQ_FILL_PERCENT=10
export FI_CXI_RETRY_COUNT=100

SYSTEMS=("cascade" "lmcache" "pdc" "llm-gpu" "hdf5-indep")
REQS=$((8 * 64)) # 512 Requests

echo "--- 📈 WEAK SCALING [64 NODES ONLY] ---"

# 1. Standard Systems
for SYS in "${SYSTEMS[@]}"; do
    echo ">> Running Weak Scaling for $SYS at 64 Nodes"
    srun --overlap -N 64 -n 64 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYS --num-requests $REQS --run-id "v11_weak_64n_${SLURM_JOB_ID}_$SYS"
    sleep 5
done

# 2. Redis Special Case
echo ">> Running Weak Scaling for lmcache-redis at 64 Nodes"
REDIS_CONF="benchmark/tmp/redis_64n_weak_${SLURM_JOB_ID}.conf"
cat << 'REOF' > $REDIS_CONF
bind 0.0.0.0
port 16379
protected-mode no
maxmemory 600gb
maxmemory-policy allkeys-lru
save ""
REOF

REDIS_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
srun --overlap -N 1 -n 1 -w $REDIS_NODE redis-server $REDIS_CONF &
SERVER_PID=$!
sleep 15
export REDIS_HOST=$REDIS_NODE
export REDIS_PORT=16379

srun --overlap -N 64 -n 64 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system lmcache-redis --num-requests $REQS --run-id "v11_weak_64n_${SLURM_JOB_ID}_redis"

srun --overlap -N 1 -n 1 -w $REDIS_NODE redis-cli -p 16379 shutdown
wait $SERVER_PID
rm $REDIS_CONF

echo "✅ Finished 64N Weak Scaling"
