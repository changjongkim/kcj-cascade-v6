#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J v5_real_data
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v5_real_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v5_real_%j.err

module load python cudatoolkit cmake gcc/12.2.0 cray-mpich

export PYTHONPATH=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build:/pscratch/sd/s/sgkim/kcj/Cascade-kcj:$PYTHONPATH

echo "Running Real App Data Benchmark..."
python3 /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/scripts/v5_real_data_test.py
