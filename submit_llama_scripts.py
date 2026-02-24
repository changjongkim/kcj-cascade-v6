import os
import subprocess

nodes_list = [1, 2, 4, 8, 16]

template = """#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q {QUEUE}
#SBATCH -t {TIME_LIMIT}
#SBATCH -N {NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_llama_70b_{NODES}n
#SBATCH -o benchmark/logs/llama_70b_{NODES}n_%j.out
#SBATCH -e benchmark/logs/llama_70b_{NODES}n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

# Ensure correct library paths for Redis and other dependencies
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Start Redis on each node
srun -n $SLURM_NNODES --ntasks-per-node=1 /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server --port 16379 --daemonize yes --maxmemory 100gb --maxmemory-policy allkeys-lru
sleep 5

echo "###################################################################"
echo " COLD START STRONG SCALING (Lustre -> GPU): Llama-3-70B"
echo "###################################################################"

srun -n {NODES} --ntasks-per-node=1 python3 benchmark/scripts/v6_contention_scaling_all.py \\
    --mode strong \\
    --data-type real \\
    --model llama-3-70b \\
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \\
    --cold

# Cleanup Redis
srun -n $SLURM_NNODES --ntasks-per-node=1 /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-cli -p 16379 shutdown nosave
"""

for nodes in nodes_list:
    queue = "regular" if nodes == 16 else "debug"
    time_limit = "00:30:00"
    
    content = template.format(
        NODES=nodes,
        QUEUE=queue,
        TIME_LIMIT=time_limit
    )
    
    filename = f"benchmark/scripts/v6_llama_70b_cold_strong_{nodes}n.slurm"
    with open(filename, "w") as f:
        f.write(content)
        
    print(f"Submitting {filename}")
    subprocess.run(["sbatch", filename])
