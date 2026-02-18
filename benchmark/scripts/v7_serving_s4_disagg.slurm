#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J v7_serv_s4
#SBATCH -o benchmark/logs/v7_serv_s4_%j.out
#SBATCH -e benchmark/logs/v7_serv_s4_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Scenario 4: Production Disaggregation (Llama-3-70B)"
srun python3 benchmark/scripts/v7_serving_at_scale.py --scenario 4 --systems Cascade,HDF5,vLLM-GPU,LMCache
