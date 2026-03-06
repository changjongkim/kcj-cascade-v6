#!/bin/bash
#SBATCH -A m1248
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J deepcam_data
#SBATCH -o benchmark/logs/gen_deepcam_%j.out

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "Starting DeepCAM Dummy Data Generation..."
srun python benchmark/scripts/generate_deepcam_dummy.py \
    --out_dir /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/deepcam_dummy_data \
    --num_train 2048 \
    --num_val 64 \
    --num_workers 64

echo "Data generation finished!"
