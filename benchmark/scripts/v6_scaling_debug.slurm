#!/bin/bash
#SBATCH --job-name=v6_dist_debug
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:05:00
#SBATCH --output=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_debug_%j.out
#SBATCH --error=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_debug_%j.err

# Set Project Directory
export PROJECT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"

# Source environment (handles modules and compiler wrappers)
source ${PROJECT_DIR}/setup_env.sh

# Ensure PYTHONPATH includes the build directory where cascade_cpp.so resides
export PYTHONPATH="${PROJECT_DIR}/cascade_Code/cpp/build_cascade_cpp:${PYTHONPATH}"

BENCH_SCRIPT="${PROJECT_DIR}/benchmark/scripts/v6_distributed_bench.py"

printf "| Nodes | Agg Write (GB/s) | Agg Read (GB/s) |\n"
printf "|-------|-----------------|-----------------|\n"

for N in 2
do
    # Run benchmark with environment sourced on each node
    srun -N $N -n $N --gpus-per-node=4 \
        bash -c "source ${PROJECT_DIR}/setup_env.sh && python $BENCH_SCRIPT" > ${PROJECT_DIR}/benchmark/logs/v6_debug_N${N}_${SLURM_JOB_ID}.log 2>&1
    
    # Extract results
    W=$(grep "Write:" ${PROJECT_DIR}/benchmark/logs/v6_debug_N${N}_${SLURM_JOB_ID}.log | tail -1 | awk '{print $2}')
    R=$(grep "Read:" ${PROJECT_DIR}/benchmark/logs/v6_debug_N${N}_${SLURM_JOB_ID}.log | tail -1 | awk '{print $2}')
    
    printf "|  %d    |      %.2f       |      %.2f       |\n" "$N" "$W" "$R"
done
