#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_rdma_adv
#SBATCH -o benchmark/logs/rdma_advanced_%j.out
#SBATCH -e benchmark/logs/rdma_advanced_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

PYTHON_BIN=$(which python3)
srun $PYTHON_BIN benchmark/scripts/v6_rdma_advanced.py
