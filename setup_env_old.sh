#!/bin/bash
# setup_env.sh - Refined for GCC 11.2 and GPU-aware MPI

# 1. Activate conda environment (FIRST, so modules can override it)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/global/common/software/nersc/python/3.9")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate kcj_qsim_mpi

# 2. Start with clean slate or standard environment
module reset
module load PrgEnv-gnu
module load gcc/11.2.0

# 4. Environment Variables
# Removed hardcoded LD_LIBRARY_PATH to avoid libfabric mismatch errors
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 5. Add Cascade Build directory to PYTHONPATH
REPO_ROOT=$(pwd)
export PYTHONPATH=$REPO_ROOT/cascade_Code/cpp/build_cascade_cpp:$PYTHONPATH
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

echo "--------------------------------------------------------"
echo "🚀 Cascade GPU-Aware MPI Environment Loaded"
echo "--------------------------------------------------------"
echo "Python: $(which python3)"
echo "C++: $(which g++) ($(g++ --version | head -n 1))"
echo "Cray CC: $(which CC)"
echo "MPI mpicc: $(which mpicc)"
echo "--------------------------------------------------------"
# Explicitly set host compiler for CUDA to avoid Conda GCC interference
export CUDAHOSTCXX=$(which g++)

# 3. Load MPI and CUDA (order matters for detecting correct MPICH version)
module load cray-mpich
module load libfabric
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load nccl/2.21.5
module load cmake
module load python/3.9
