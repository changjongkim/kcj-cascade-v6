#!/bin/bash
# setup_env.sh - Final robust version for Perlmutter

# 1. Start with clean slate
module reset || true

# 2. Activate conda environment FIRST
module load python/3.9
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/global/common/software/nersc/python/3.9")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate kcj_qsim_mpi

# 3. Load Perlmutter Modules AFTER conda
# This ensures srun/cc/CC and Cray libs take precedence
module load PrgEnv-gnu
module load gcc-native/12.3
module load cray-mpich
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load cmake

# 4. Critical Linker & Execution Paths
# Force use of newer GCC libs (required by cray-mpich 8.1.30)
# We use LD_LIBRARY_PATH here but will use LD_PRELOAD at runtime if needed
export LD_LIBRARY_PATH=/opt/cray/pe/gcc/12.2.0/snos/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/pe/lib64:$LD_LIBRARY_PATH

# 5. Environment Variables
export CUDAHOSTCXX=$(which g++)
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CRAY_CPU_TARGET=x86-64

# 6. Add Cascade Build directory to PYTHONPATH
REPO_ROOT=$(pwd)
export PYTHONPATH=$REPO_ROOT/cascade_Code/cpp/build:$PYTHONPATH
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

echo "--------------------------------------------------------"
echo "🚀 Cascade GPU-Aware MPI Environment Loaded"
echo "--------------------------------------------------------"
echo "Python: $(which python3)"
echo "C++: $(which g++) ($(g++ --version | head -n 1))"
echo "Cray CC: $(which CC)"
echo "--------------------------------------------------------"
