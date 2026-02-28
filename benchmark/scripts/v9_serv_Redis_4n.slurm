#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_Redis_4n
#SBATCH -o benchmark/logs/v9_Redis_4n_%j.out
#SBATCH -e benchmark/logs/v9_Redis_4n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Start Redis on each node
echo "Starting Redis server on each node..."
srun --overlap -n $SLURM_NNODES --ntasks-per-node=1 bash -c "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server --port 16379 --bind 0.0.0.0 --protected-mode no --maxmemory 100gb --maxmemory-policy allkeys-lru" &
sleep 30

srun --overlap -n 4 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system LMCache-Redis \
    --num-requests 32 \
    --gen-tokens 32

# Cleanup Redis
srun --overlap -n $SLURM_NNODES --ntasks-per-node=1 bash -c "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-cli -p 16379 shutdown nosave"
pkill -u $USER redis-server
