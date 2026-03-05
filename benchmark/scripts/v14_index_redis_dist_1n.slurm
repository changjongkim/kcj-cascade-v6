#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J idx_redis_dist_1n
#SBATCH -o benchmark/logs/v14_idx_redis_dist__1n_%j.out
#SBATCH -e benchmark/logs/v14_idx_redis_dist__1n_%j.err

# Load modules
module load cudatoolkit/12.4
module load gcc-native/12.3

# Setup paths
export REPO_ROOT=/pscratch/sd/s/sgkim/kcj/Cascade-kcj
cd $REPO_ROOT

SYSTEM="redis-dist"
PORT=16379

# Create host file directory
JOB_TMP_DIR=$REPO_ROOT/benchmark/tmp/hosts_$SLURM_JOB_ID
mkdir -p $JOB_TMP_DIR

# Get all hostnames and save to file
scontrol show hostnames $SLURM_JOB_NODELIST > $JOB_TMP_DIR/all_hosts

echo "Starting Redis servers on all nodes..."
# Start redis-server on EACH node in background
srun --overlap -N 1 -n 1 --ntasks-per-node=1 \
    redis-server --port $PORT --maxmemory 100gb --maxmemory-policy allkeys-lru --protected-mode no --save "" --appendonly no &
REDIS_SRUN_PID=$!

sleep 10 # Wait for all servers to start

# Export for adapter
export REDIS_PORT=$PORT

echo "=== Index Scalability: Distributed Sharded Redis ==="
srun --overlap -N 1 -n 1 --ntasks-per-node=1 \
    python3 benchmark/scripts/v14_index_scalability_bench4.py \
    --system $SYSTEM \
    --test-reqs 128 \
    --block-size 16 --scale-steps 50000

# Cleanup
pkill -u $USER redis-server
wait $REDIS_SRUN_PID 2>/dev/null

echo "✅ Done"
