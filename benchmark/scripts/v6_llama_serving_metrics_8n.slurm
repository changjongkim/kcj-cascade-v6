#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_serv_metrics_8n
#SBATCH -o benchmark/logs/serv_metrics_8n_%j.out
#SBATCH -e benchmark/logs/serv_metrics_8n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Force Conda library path for Redis GLIBCXX 3.4.32
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Start Redis on each node for LMCache-Redis
echo "Starting Redis server on each node..."
srun --overlap -n $SLURM_NNODES --ntasks-per-node=1 bash -c "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server --port 16379 --maxmemory 100gb --maxmemory-policy allkeys-lru" &
sleep 20

echo "###################################################################"
echo " LLM Serving Metrics Benchmark: Llama-3-70B 8 Nodes"
echo "###################################################################"

# Run the metrics benchmark
# We use --overlap to allow client and server on same node if needed
# But here we have 4 GPUs, so we can run 4 tasks per node if we want to simulate concurrent requests better.
# For simplicity, we stick to N tasks total (1 per node) as in our scaling benchmarks, 
# but each task will handle multiple requests sequentially or concurrently.

srun --overlap -n 8 --ntasks-per-node=1 python3 benchmark/scripts/v6_serving_metrics_llama.py \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache,LMCache-Redis \
    --num-requests 16 \
    --gen-tokens 64

# Cleanup Redis
echo "Cleaning up Redis servers..."
srun --overlap -n $SLURM_NNODES --ntasks-per-node=1 bash -c "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-cli -p 16379 shutdown nosave"
pkill -u $USER redis-server
