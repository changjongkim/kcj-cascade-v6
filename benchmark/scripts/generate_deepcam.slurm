#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J deepcam_data_512gb
#SBATCH -o benchmark/logs/gen_deepcam_512gb_%j.out

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source ml_workspace/setup_ml_isolated.sh

echo "Starting DeepCAM 512GB Dummy Data Generation..."
srun python benchmark/scripts/generate_deepcam_dummy.py \
    --out_dir /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/deepcam_dummy_data_512gb \
    --num_train 9263 \
    --num_val 128 \
    --num_workers 64

echo "Data generation finished!"
