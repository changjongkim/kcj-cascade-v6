#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_ensemble_n2
#SBATCH -o benchmark/logs/ensemble_n2_%j.out
#SBATCH -e benchmark/logs/ensemble_n2_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Running ensemble with 2 nodes (debug queue)"
srun python3 benchmark/scripts/v6_app_ensemble.py --block-size 128 --ic-blocks 10 --member-blocks 5 --evolution-steps 5 --exchange-rounds 3 --systems Cascade,HDF5,POSIX,vLLM-GPU,PDC,LMCache
