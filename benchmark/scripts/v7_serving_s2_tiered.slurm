#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J v7_serv_s2
#SBATCH -o benchmark/logs/v7_serv_s2_%j.out
#SBATCH -e benchmark/logs/v7_serv_s2_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Scenario 2: Full-Tiered Stress (Llama-3-70B)"
srun python3 benchmark/scripts/v7_serving_at_scale.py --scenario 2 --systems Cascade,HDF5,vLLM-GPU,LMCache
