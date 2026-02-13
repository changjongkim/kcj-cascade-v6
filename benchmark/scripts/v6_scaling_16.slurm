#!/bin/bash
#SBATCH --job-name=v6_scale_16
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_scale_16_%j.out
#SBATCH --error=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_scale_16_%j.err

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

N=16
echo "Running benchmark on $N nodes..."
srun -N $N -n $N --gpus-per-node=4 python $BENCH_SCRIPT > ${PROJECT_DIR}/benchmark/logs/v6_scale_N${N}_${SLURM_JOB_ID}.log 2>&1

# Extract results
W=$(grep "Write:" ${PROJECT_DIR}/benchmark/logs/v6_scale_N${N}_${SLURM_JOB_ID}.log | tail -1 | awk '{print $2}')
R=$(grep "Read:" ${PROJECT_DIR}/benchmark/logs/v6_scale_N${N}_${SLURM_JOB_ID}.log | tail -1 | awk '{print $2}')

echo "  $N   |     $W         |      $R"
