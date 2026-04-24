#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v16_dedup_sens_redis_8n
#SBATCH -o benchmark/logs/v16_dedup_sens_redis_8n_%j.out

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

RATES="0.1 0.3 0.5 0.7"
SYSTEM="redis"
BLOCK_SIZE=320
NUM_BLOCKS=64

for rate in $RATES; do
    echo ">>>> Running $SYSTEM with Sharing Rate: $rate <<<<"
    srun -N 8 -n 8 python3 benchmark/scripts/v16_dedup_sensitivity_bench.py \
        --system $SYSTEM \
        --sharing-rate $rate \
        --num-blocks $NUM_BLOCKS \
        --block-size-mb $BLOCK_SIZE \
        --cleanup
done
