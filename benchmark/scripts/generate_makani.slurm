#!/bin/bash
#SBATCH -A m1248
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J makani_data
#SBATCH -o benchmark/logs/gen_makani_%j.out

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "Starting Makani Dummy Data Generation..."
# Generate 1460 samples (365 days * 4 per day)
# Resolution: 720x1440, default channels: 20
srun python benchmark/scripts/generate_makani_dummy.py \
    --out_dir /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/makani_dummy_data \
    --num_samples_per_year 1460 \
    --num_channels 20 \
    --img_size_h 720 \
    --img_size_w 1440 \
    --num_workers 3

echo "Data generation finished!"
