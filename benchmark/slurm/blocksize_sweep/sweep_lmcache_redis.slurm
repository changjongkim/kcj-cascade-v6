#!/bin/bash
#SBATCH -J bs_sweep_lmr
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -A m5320_g
#SBATCH -o benchmark/logs/sweep_lmcache_redis_%j.out

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

# Start Redis on the head node
REDIS_CONF="benchmark/tmp/redis_sweep_${SLURM_JOB_ID}.conf"
cat << REOF > $REDIS_CONF
bind 0.0.0.0
port 16379
protected-mode no
maxmemory 400gb
maxmemory-policy allkeys-lru
save ""
REOF

REDIS_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "🚀 Starting Redis on $REDIS_NODE at port 16379"
srun --overlap -N 1 -n 1 -w $REDIS_NODE redis-server $REDIS_CONF &
SERVER_PID=$!
sleep 15
export REDIS_HOST=$REDIS_NODE
export REDIS_PORT=16379

for t in $TOKENS; do
    echo "[Tokens: $t]"
    srun -N 8 -n 8 python3 -m benchmark.run_benchmark \
        --systems lmcache-redis \
        --workload throughput \
        --num_blocks 128 \
        --block_size_tokens $t
done

# Cleanup
echo "Shutting down Redis..."
srun --overlap -N 1 -n 1 -w $REDIS_NODE redis-cli -p 16379 shutdown
wait $SERVER_PID
rm $REDIS_CONF
echo "✅ Finished Sweep"
