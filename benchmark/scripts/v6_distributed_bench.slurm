#!/bin/bash
#SBATCH --job-name=v6_dist_bench
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00
#SBATCH --output=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_dist_%j.out
#SBATCH --error=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_dist_%j.err

# ============================================================================
#  Cascade V6 Distributed Benchmark — 3 Novelties
#  Tests: Dedup, Semantic Eviction, Locality-Aware Placement, Scaling
# ============================================================================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Cascade V6 Distributed Benchmark                  ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Job ID:    %-45s\n" "$SLURM_JOB_ID"
printf "║  Nodes:     %-45s\n" "$SLURM_NNODES"
printf "║  Tasks:     %-45s\n" "$SLURM_NTASKS"
printf "║  Nodelist:  %-45s\n" "$SLURM_NODELIST"
printf "║  Timestamp: %-45s\n" "$(date +%Y%m%d_%H%M%S)"
echo "╚══════════════════════════════════════════════════════════╝"

# Environment setup
module load PrgEnv-gnu 2>/dev/null
module load gcc-native/13.2 2>/dev/null
module load cudatoolkit/12.4 2>/dev/null
module load cray-python 2>/dev/null
module load cray-mpich 2>/dev/null

export MPICH_GPU_SUPPORT_ENABLED=1
export PROJECT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
export PYTHONPATH="${PROJECT_DIR}/cascade_Code/cpp/build_mpi:${PYTHONPATH}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Running V6 Distributed Benchmark (Python + MPI)"
echo "═══════════════════════════════════════════════════════════"

srun python ${PROJECT_DIR}/benchmark/scripts/v6_distributed_bench.py

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  V6 Distributed Benchmark Complete"
echo "═══════════════════════════════════════════════════════════"
