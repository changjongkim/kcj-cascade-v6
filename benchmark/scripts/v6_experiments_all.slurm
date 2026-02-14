#!/bin/bash
#SBATCH --job-name=v6_full_exp
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_full_exp_%j.out
#SBATCH --error=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_full_exp_%j.err

# ============================================================================
#  Cascade V6 Full Experiments: 1, 2, 4, 8 Nodes Data Sweeps
# ============================================================================

module load PrgEnv-gnu 2>/dev/null
module load gcc-native/13.2 2>/dev/null
module load cudatoolkit/12.4 2>/dev/null
module load cray-python 2>/dev/null
module load cray-mpich 2>/dev/null

export MPICH_GPU_SUPPORT_ENABLED=1
export PROJECT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
export PYTHONPATH="${PROJECT_DIR}/cascade_Code/cpp/build_mpi:${PYTHONPATH}"

BENCH_SCRIPT="${PROJECT_DIR}/benchmark/scripts/v6_full_suite.py"

echo "======================================================================"
echo "STARTING FULL EXPERIMENTAL SUITE (1, 2, 4, 8 Nodes)"
echo "Target: SC26 Comprehensive Evaluation"
echo "======================================================================"

# Function to run benchmark for N nodes
run_bench() {
    local N=$1
    echo ""
    echo "------------------------------------------------------------------"
    echo "Running Full Suite on $N Nodes..."
    echo "------------------------------------------------------------------"
    srun -N $N -n $N --gpus-per-node=4 python $BENCH_SCRIPT
}

# Run sequential sweeps
run_bench 1
run_bench 2
run_bench 4
run_bench 8

echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETED."
echo "======================================================================"
