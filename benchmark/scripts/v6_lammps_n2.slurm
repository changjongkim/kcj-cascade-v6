#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J v6_lammps_n2
#SBATCH -D /pscratch/sd/s/sgkim/kcj/Cascade-kcj
#SBATCH -o benchmark/logs/lammps_n2_%j.out
#SBATCH -e benchmark/logs/lammps_n2_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Start Redis on each node
srun -n $SLURM_NNODES --ntasks-per-node=1 /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server --port 16379 --daemonize yes --maxmemory 100gb --maxmemory-policy allkeys-lru
sleep 2
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

srun -n 2 --ntasks-per-node=1 -c 32 --gpus-per-node=4 \
    python benchmark/scripts/v6_app_lammps_io.py \
    --atoms_per_rank 100000000 \
    --timesteps 5 \
    --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache,Redis

# Cleanup Redis
srun -n $SLURM_NNODES --ntasks-per-node=1 /pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-cli -p 16379 shutdown nosave
