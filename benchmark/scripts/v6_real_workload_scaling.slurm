#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=8
#SBATCH --account=m1248_g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00
#SBATCH --job-name=v6_real_scaling
#SBATCH --output=%x_%j.out

# Project Setup
PROJECT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
BENCH_SCRIPT="${PROJECT_DIR}/benchmark/scripts/v6_real_workload_bench.py"
LOG_DIR="${PROJECT_DIR}/benchmark/logs"
mkdir -p $LOG_DIR

# Environment Setup
source ${PROJECT_DIR}/setup_env.sh

# MPI and Performance Optimization
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_ATS=0
export FI_CXI_OPTIM_GET=1

echo "Starting Cascade V6 Real Workload Scaling Study..."
echo "Nodes: 1, 2, 4, 8"

for N in 1 2 4 8
do
    echo "--------------------------------------------------------"
    echo "Running benchmark on $N nodes..."
    echo "--------------------------------------------------------"
    
    # Run the benchmark
    # Use -n $N --gpus-per-node=4 (assuming 1 MPI rank per node, which manages 4 GPUs)
    srun -N $N -n $N --gpus-per-node=4 \
        bash -c "source ${PROJECT_DIR}/setup_env.sh && python $BENCH_SCRIPT" > ${LOG_DIR}/v6_real_N${N}_${SLURM_JOB_ID}.log 2>&1
    
    echo "Done. Results logged to ${LOG_DIR}/v6_real_N${N}_${SLURM_JOB_ID}.log"
done

echo "Real Workload Scaling Study Complete."
