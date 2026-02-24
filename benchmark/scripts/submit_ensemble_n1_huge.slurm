#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_ensemble_n1_huge
#SBATCH -o benchmark/logs/ensemble_n1_huge_%j.out
#SBATCH -e benchmark/logs/ensemble_n1_huge_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Running HUGE SCALE ensemble with 1 nodes (regular queue)"
srun python3 benchmark/scripts/v6_app_ensemble.py --block-size 1024 --ic-blocks 100 --member-blocks 10 --evolution-steps 6 --exchange-rounds 3 --systems Cascade,HDF5,POSIX,vLLM-GPU,PDC,LMCache
