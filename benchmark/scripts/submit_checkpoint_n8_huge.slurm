#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_checkpoint_n8_huge
#SBATCH -o benchmark/logs/checkpoint_n8_huge_%j.out
#SBATCH -e benchmark/logs/checkpoint_n8_huge_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Running HUGE SCALE checkpoint with 8 nodes (regular queue)"
srun python3 benchmark/scripts/v6_app_checkpoint.py --mode checkpoint --field-size 2048 --num-fields 5 --timesteps 16 --ckpt-interval 2 --delta-pct 5 --systems Cascade,HDF5,POSIX,vLLM-GPU,PDC,LMCache
