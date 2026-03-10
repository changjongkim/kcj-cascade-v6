#!/bin/bash
# ml_workspace/setup_ml_isolated.sh - FULLY ISOLATED environment for DeepCAM/Makani

# 1. Start with clean slate
module reset || true

# 2. Activate ML-specific conda environment (Python 3.11)
module load python
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate /pscratch/sd/s/sgkim/kcj/Cascade-kcj/conda_env

# 3. Load Perlmutter Modules
module load PrgEnv-gnu
module load gcc-native/12.3
module load cray-mpich
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load cmake

# 4. Critical Environment Variables
module unload darshan
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDAHOSTCXX=$(which g++)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CRAY_CPU_TARGET=x86-64

# Cleanup conflicting MPI variables for the custom conda env
unset MPICH_OFI_NIC_POLICY
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1

# Networking for Torch Distributed on Perlmutter Slingshot
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1

ML_ROOT="/pscratch/sd/s/sgkim/kcj/Cascade-kcj/ml_workspace"
REPO_ROOT="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"

# Use the ML-specific build directory
export PYTHONPATH=$ML_ROOT/build_ml_311:$PYTHONPATH
export PYTHONPATH=$ML_ROOT/makani:$PYTHONPATH
export PYTHONPATH=$ML_ROOT:$PYTHONPATH
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

echo "--------------------------------------------------------"
echo "🌐 ISOLATED ML Environment (Python 3.11) Loaded"
echo "Python: $(which python)"
echo "Build Dir: $ML_ROOT/build_ml_311"
echo "--------------------------------------------------------"
