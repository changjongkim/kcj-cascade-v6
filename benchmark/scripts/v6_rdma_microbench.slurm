#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_rdma_bench
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/rdma_bench_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/rdma_bench_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

PYTHON_BIN=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/python3
SCRIPT=benchmark/scripts/v6_rdma_microbench.py

echo "Starting RDMA Micro-benchmarks Sweep (1, 2, 4, 8 Nodes)"

for N in 1 2 4 8; do
    echo ""
    echo ">>>> Running with N=$N Nodes <<<<"
    srun -N $N -n $N --gpus-per-node=4 $PYTHON_BIN $SCRIPT
done
