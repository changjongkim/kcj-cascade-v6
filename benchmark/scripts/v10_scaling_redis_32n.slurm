#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v10_redis_32n
#SBATCH -o benchmark/logs/v10_redis_32n_%j.out
#SBATCH -e benchmark/logs/v10_redis_32n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "Starting Redis on 32 nodes..."
srun --overlap -N 32 -n 32 --ntasks-per-node=1 redis-server --port 16379 --bind 0.0.0.0 --protected-mode no --save "" &
sleep 30

SYSTEM="lmcache-redis"
NODES_LIST=(1 2 4 8 16 32)
JOB_UID=${{SLURM_JOB_ID}}

echo "🚀 Starting Redis Scaling (1-32 nodes)"

for N in "${{NODES_LIST[@]}}"; do
    echo "   [Scale: $N Nodes]"
    REQS=$((8 * N))
    RUN_ID="${{JOB_UID}}_N${{N}}"
    
    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py         --system $SYSTEM         --num-requests $REQS         --run-id $RUN_ID
    
    sleep 5
done

# Cleanup Redis
pkill -u $USER redis-server
echo "✅ Finished Redis"
