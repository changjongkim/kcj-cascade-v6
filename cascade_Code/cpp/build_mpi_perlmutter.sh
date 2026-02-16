#!/bin/bash
#
# Build script for Cascade C++ with MPI on Perlmutter
# Uses Cray compiler wrappers (CC/cc) for MPI support
#

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Cascade C++ MPI Build for Perlmutter                   ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Load required modules
module load PrgEnv-gnu 2>/dev/null || true
module load gcc-native/12.3 2>/dev/null || true
module load cudatoolkit/12.4 2>/dev/null || true
module load cmake/3.24 2>/dev/null || true
module load cray-python 2>/dev/null || true
module load cray-mpich 2>/dev/null || true

# Set MPICH_GPU to enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80

# Install pybind11 if needed
pip show pybind11 >/dev/null 2>&1 || pip install --user pybind11

# Create MPI build directory (separate from non-MPI build)
BUILD_DIR="build_mpi"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake using Cray CC wrapper for MPI
echo ""
echo "Configuring with CMake (MPI enabled)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPERLMUTTER=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CUDA_HOST_COMPILER=CC \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DUSE_MPI=ON \
    -DBUILD_PYTHON=ON

# Build
echo ""
echo "Building (MPI + CUDA)..."
make -j$(nproc)

# Copy outputs
echo ""
echo "Installing..."
cp cascade_cpp*.so ../ 2>/dev/null || true
cp cascade_bench ../ 2>/dev/null || true
cp distributed_bench ../ 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Build complete (MPI enabled)!                          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Python:  cascade_cpp.cpython-*.so  (with MPI)          ║"
echo "║  C++:     cascade_bench, distributed_bench              ║"
echo "║                                                          ║"
echo "║  Test (2 nodes, 4 GPUs each):                           ║"
echo "║    srun -N2 -n8 --gpus-per-node=4 python bench.py       ║"
echo "╚══════════════════════════════════════════════════════════╝"
