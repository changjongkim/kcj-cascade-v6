#!/bin/bash
#SBATCH --job-name=v6_scale_small
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_small_%j.out
#SBATCH --error=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_small_%j.err

# ============================================================================
#  Cascade V6 Small Scale Study: 1, 2, 4, 8 Nodes
# ============================================================================

# Ensure environment matches build environment
echo "[DEBUG] Starting SLURM script..."
echo "[DEBUG] SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
source ${SLURM_SUBMIT_DIR}/setup_env.sh
echo "[DEBUG] Environment Loaded."

export MPICH_GPU_SUPPORT_ENABLED=1
export PROJECT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
export CASCADE_BUILD_DIR="${PROJECT_DIR}/cascade_Code/cpp/build_clean"
# Point to the clean build directory
export PYTHONPATH="${PROJECT_DIR}/cascade_Code/cpp/build_clean:${PYTHONPATH}"

BENCH_SCRIPT="${PROJECT_DIR}/benchmark/scripts/v6_distributed_bench.py"

echo "Nodes | Agg Write (GB/s) | Agg Read (GB/s)"
echo "------------------------------------------"

for N in 1 2 4 8
do
    echo "Running benchmark on $N nodes..."
    # We use -n $N to have 1 task per node, each task managing its local GPUs
    srun -N $N -n $N --gpus-per-node=4 python -u $BENCH_SCRIPT > ${PROJECT_DIR}/benchmark/logs/v6_small_N${N}_${SLURM_JOB_ID}.log 2>&1
    
    # Extract aggregate throughput from Test 4
    W=$(grep "Write:" ${PROJECT_DIR}/benchmark/logs/v6_small_N${N}_${SLURM_JOB_ID}.log | tail -1 | awk '{print $2}')
    R=$(grep "Read:" ${PROJECT_DIR}/benchmark/logs/v6_small_N${N}_${SLURM_JOB_ID}.log | tail -1 | awk '{print $2}')
    
    echo "  $N   |     $W         |      $R"
done

echo "------------------------------------------"
echo "Small scale study complete."
