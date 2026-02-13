#!/bin/bash
#SBATCH --job-name=v6_scale_large
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_large_%j.out
#SBATCH --error=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_large_%j.err

# ============================================================================
#  Cascade V6 Large Scale Study: 16, 32, 64 Nodes
# ============================================================================

module load PrgEnv-gnu 2>/dev/null
module load gcc-native/13.2 2>/dev/null
module load cudatoolkit/12.4 2>/dev/null
module load cray-python 2>/dev/null
module load cray-mpich 2>/dev/null

export MPICH_GPU_SUPPORT_ENABLED=1
export PROJECT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
export PYTHONPATH="${PROJECT_DIR}/cascade_Code/cpp/build_mpi:${PYTHONPATH}"

BENCH_SCRIPT="${PROJECT_DIR}/benchmark/scripts/v6_distributed_bench.py"

echo "Nodes | Agg Write (GB/s) | Agg Read (GB/s)"
echo "------------------------------------------"

for N in 16 32 64
do
    echo "Running benchmark on $N nodes..."
    # We use -n $N to have 1 task per node, each task managing its local GPUs
    srun -N $N -n $N --gpus-per-node=4 python $BENCH_SCRIPT > ${PROJECT_DIR}/benchmark/logs/v6_large_N${N}_${SLURM_JOB_ID}.log 2>&1
    
    # Extract aggregate throughput from Test 4
    W=$(grep "Write:" ${PROJECT_DIR}/benchmark/logs/v6_large_N${N}_${SLURM_JOB_ID}.log | tail -1 | awk '{print $2}')
    R=$(grep "Read:" ${PROJECT_DIR}/benchmark/logs/v6_large_N${N}_${SLURM_JOB_ID}.log | tail -1 | awk '{print $2}')
    
    echo "  $N   |     $W         |      $R"
done

echo "------------------------------------------"
echo "Large scale study complete."
