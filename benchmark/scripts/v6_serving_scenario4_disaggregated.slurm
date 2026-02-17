#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J serv_s4_disagg
#SBATCH -o benchmark/logs/serv_s4_disagg_%j.out
#SBATCH -e benchmark/logs/serv_s4_disagg_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Scenario 4: Disaggregated Stress (interference)"
srun python3 benchmark/scripts/v6_serving_benchmarks.py --scenario 4 --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache
