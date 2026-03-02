#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v10_str_32n_redis
#SBATCH -o benchmark/logs/v10_str_32n_redis_%j.out
#SBATCH -e benchmark/logs/v10_str_32n_redis_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh



SYSTEM="lmcache-redis"
NODES_LIST=(16 32)
JOB_UID=${SLURM_JOB_ID}

echo "🚀 Starting redis Strong Scaling (16-32 nodes, 128 Reqs Total)"


# Start Redis Server on Rank 0
REDIS_CONF="benchmark/tmp/redis_str_32n_${JOB_UID}.conf"
cat << 'REOF' > $REDIS_CONF
bind 0.0.0.0
port 16379
protected-mode no
maxmemory 600gb
maxmemory-policy allkeys-lru
save ""
appendonly no
REOF

REDIS_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Starting Redis on $REDIS_NODE at port 16379"
srun --overlap -N 1 -n 1 -w $REDIS_NODE redis-server $REDIS_CONF &
SERVER_PID=$!
sleep 15
export REDIS_HOST=$REDIS_NODE
export REDIS_PORT=16379


for N in "${NODES_LIST[@]}"; do
    echo "   [Scale: $N Nodes]"
    REQS=128
    RUN_ID="${JOB_UID}_N${N}"
    
    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID
    
    sleep 5
done


echo "Shutting down Redis..."
srun --overlap -N 1 -n 1 -w $REDIS_NODE redis-cli -h $REDIS_NODE -p 6379 shutdown
wait $SERVER_PID
rm -f $REDIS_CONF


echo "✅ Finished redis"
