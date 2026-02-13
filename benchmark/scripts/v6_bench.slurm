#!/bin/bash
#SBATCH --job-name=v6_bench
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_bench_%j.out
#SBATCH --error=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v6_bench_%j.err

# ============================================================================
#  Cascade V6 Comprehensive Benchmark
#  Tests: Synthetic + Real App Data + Feature Ablation
# ============================================================================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Cascade V6 Comprehensive Benchmark                ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Job ID:    %-45s\n" "$SLURM_JOB_ID"
printf "║  Node:      %-45s\n" "$SLURMD_NODENAME"
printf "║  Timestamp: %-45s\n" "$(date +%Y%m%d_%H%M%S)"
printf "║  GPU:       %-45s\n" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "╚══════════════════════════════════════════════════════════╝"

# Environment setup
module load PrgEnv-gnu 2>/dev/null
module load gcc-native/13.2 2>/dev/null
module load cudatoolkit/12.4 2>/dev/null
module load cray-python 2>/dev/null

export PROJECT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
export PYTHONPATH="${PROJECT_DIR}/cascade_Code/cpp/build:${PROJECT_DIR}:${PYTHONPATH}"

# ============================================================================
#  Part 1: Raw C++ Backend Throughput (cascade_bench)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Part 1: Raw Backend Throughput (cascade_bench C++)"
echo "═══════════════════════════════════════════════════════════"

BENCH_BIN="${PROJECT_DIR}/cascade_Code/cpp/build/cascade_bench"

if [ -f "$BENCH_BIN" ]; then
    echo ""
    echo "--- GPU + SHM (1MB blocks x 500, 8 threads) ---"
    $BENCH_BIN --block-size 1048576 --num-blocks 500 --threads 8 --gpu-capacity 600 --shm-capacity 600

    echo ""
    echo "--- GPU + SHM (128KB blocks x 1000, 8 threads) ---"
    $BENCH_BIN --block-size 131072 --num-blocks 1000 --threads 8 --gpu-capacity 150 --shm-capacity 150
else
    echo "WARNING: cascade_bench binary not found, skipping C++ benchmark"
fi

# ============================================================================
#  Part 2: V6 Python Comprehensive Benchmark
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Part 2: V6 Feature Benchmark (Python)"
echo "═══════════════════════════════════════════════════════════"

python ${PROJECT_DIR}/benchmark/scripts/v6_bench.py

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  V6 Benchmark Job Complete"
echo "═══════════════════════════════════════════════════════════"
