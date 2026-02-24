
import os
import subprocess

template = """#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_llama_redis_{nodes}n
#SBATCH -o benchmark/logs/llama_redis_{nodes}n_%j.out
#SBATCH -e benchmark/logs/llama_redis_{nodes}n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Force Conda library path for Redis GLIBCXX 3.4.32
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Start Redis on each node - run in background without daemonize
echo "Starting Redis server on each node..."
srun -n $SLURM_NNODES --ntasks-per-node=1 bash -c "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server --port 16379 --maxmemory 100gb --maxmemory-policy allkeys-lru" &
sleep 20

echo "###################################################################"
echo " LMCache-Redis (Redis): Llama-3-70B {nodes} Nodes"
echo "###################################################################"

srun -n {nodes} --ntasks-per-node=1 python3 benchmark/scripts/v6_contention_scaling_all.py \\
    --mode strong \\
    --data-type real \\
    --model llama-3-70b \\
    --systems LMCache-Redis \\
    --cold

# Cleanup Redis
echo "Cleaning up Redis servers..."
srun -n $SLURM_NNODES --ntasks-per-node=1 bash -c "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-cli -p 16379 shutdown nosave"
pkill -u $USER redis-server
"""

node_counts = [1, 2, 4, 8, 16]

for n in node_counts:
    filename = f"benchmark/scripts/v6_llama_redis_only_{n}n.slurm"
    with open(filename, "w") as f:
        f.write(template.format(nodes=n))
    
    # Use regular queue for 16n
    if n == 16:
        subprocess.run(["sed", "-i", "s/-q debug/-q regular/", filename])
        
    print(f"Submitting {filename}")
    subprocess.run(["sbatch", filename])
