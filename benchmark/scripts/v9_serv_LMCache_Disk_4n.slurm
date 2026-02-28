#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_LMCache_Disk_4n
#SBATCH -o benchmark/logs/v9_LMCache_Disk_4n_%j.out
#SBATCH -e benchmark/logs/v9_LMCache_Disk_4n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

srun --overlap -n 4 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system LMCache \
    --num-requests 32 \
    --gen-tokens 32
