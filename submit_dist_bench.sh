#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q interactive
#SBATCH -N 2
#SBATCH -t 00:10:00
#SBATCH -J cascade_multi_node
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/dist_bench_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/dist_bench_%j.err

mkdir -p /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs

# Load environment
module load cudatoolkit
module load cray-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONPATH=$PYTHONPATH:$(pwd)/cascade_Code/cpp/build_cascade_cpp
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH

source setup_env.sh

echo "Starting Multi-Node Benchmark on $SLURM_NNODES nodes"
srun -N 2 -n 2 --gpus-per-node=1 python3 bench_multi_node.py
