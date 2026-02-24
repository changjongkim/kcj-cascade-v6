#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_ensemble_n1_large
#SBATCH -o benchmark/logs/ensemble_n1_large_%j.out
#SBATCH -e benchmark/logs/ensemble_n1_large_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Running LARGE SCALE ensemble with 1 nodes (regular queue)"
srun python3 benchmark/scripts/v6_app_ensemble.py --block-size 512 --ic-blocks 20 --member-blocks 5 --evolution-steps 5 --exchange-rounds 3 --systems Cascade,HDF5,POSIX,vLLM-GPU,PDC,LMCache
