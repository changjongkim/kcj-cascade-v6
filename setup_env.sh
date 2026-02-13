#!/bin/bash
# setup_env.sh - Refined for GCC 11.2 and GPU-aware MPI

# 1. Start with clean slate or standard environment
module load PrgEnv-gnu 2>/dev/null
module load gcc/11.2.0 2>/dev/null

# 2. Load MPI and CUDA (order matters for detecting correct MPICH version)
module load cray-mpich 2>/dev/null
module load cudatoolkit/12.4 2>/dev/null
module load craype-accel-nvidia80 2>/dev/null
module load nccl/2.21.5 2>/dev/null
module load python/3.9 2>/dev/null

# 3. Activate conda environment
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/global/common/software/nersc/python/3.9")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate kcj_qsim_mpi

# 4. Environment Variables
export MPI_HOME=/global/u2/s/sgkim/.conda/envs/kcj_qsim_mpi
export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
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
