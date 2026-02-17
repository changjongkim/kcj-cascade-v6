#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J serv_s2_eviction
#SBATCH -o benchmark/logs/serv_s2_eviction_%j.out
#SBATCH -e benchmark/logs/serv_s2_eviction_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Scenario 2: Semantic Eviction (Prefix Protection)"
srun python3 benchmark/scripts/v6_serving_benchmarks.py --scenario 2 --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache
