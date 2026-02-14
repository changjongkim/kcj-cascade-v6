#!/bin/bash
# run_interactive_test.sh

# 1. Setup Environment
source setup_env.sh

# 2. Ensure PYTHONPATH is correct
export PROJECT_DIR=$(pwd)
export PYTHONPATH="${PROJECT_DIR}/cascade_Code/cpp/build_cascade_cpp:${PYTHONPATH}"

echo "================================================================"
echo " Starting Interactive Distributed Benchmark (v6)"
echo " Key Features: Semantic Eviction, Dedup, Locality-Awareness"
echo "================================================================"

# 3. Run Benchmark commands
# Check if inside an allocation
if [ -z "$SLURM_JOB_ID" ]; then
    echo "❌ Error: This script must be run inside a SLURM allocation (salloc/sbatch)."
    echo "   Run 'salloc -N 2 -C gpu -q debug -t 30:00 -A m1248_g' first."
    exit 1
fi

echo "Running on $SLURM_JOB_NUM_NODES nodes..."

# Run the python benchmark
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES --gpus-per-node=4 \
    python benchmark/scripts/v6_distributed_bench.py

echo "================================================================"
echo " Done."
echo "================================================================"
