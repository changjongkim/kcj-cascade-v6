#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_checkpoint_n1
#SBATCH -o benchmark/logs/checkpoint_n1_%j.out
#SBATCH -e benchmark/logs/checkpoint_n1_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Running checkpoint with 1 nodes (debug queue)"
srun python3 benchmark/scripts/v6_app_checkpoint.py --mode checkpoint --field-size 256 --num-fields 3 --timesteps 10 --ckpt-interval 5 --delta-pct 5 --systems Cascade,HDF5,POSIX,vLLM-GPU,PDC,LMCache
