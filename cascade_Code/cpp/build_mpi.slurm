#!/bin/bash
#SBATCH --job-name=cascade_mpi_build
#SBATCH --account=m1248_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --output=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/mpi_build_%j.out
#SBATCH --error=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/mpi_build_%j.err

echo "═══════════════════════════════════════════════════════════"
echo "  Building Cascade C++ with MPI on compute node"
echo "═══════════════════════════════════════════════════════════"

module load PrgEnv-gnu 2>/dev/null || true
module load gcc-native/13.2 2>/dev/null || true
module load cudatoolkit/12.4 2>/dev/null || true
module load cmake/3.24 2>/dev/null || true
module load cray-python 2>/dev/null || true
module load cray-mpich 2>/dev/null || true

export MPICH_GPU_SUPPORT_ENABLED=1

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp

mkdir -p build_mpi
cd build_mpi

echo "Configuring with CMake (MPI enabled via CC wrapper)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPERLMUTTER=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DUSE_MPI=ON \
    -DBUILD_PYTHON=ON

echo ""
echo "Building (MPI + CUDA)..."
make -j32 VERBOSE=1

echo ""
echo "Build artifacts:"
ls -la cascade_cpp*.so distributed_bench 2>/dev/null

# Copy outputs
cp cascade_cpp*.so ../ 2>/dev/null || true
cp distributed_bench ../ 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  MPI Build Complete!"
echo "═══════════════════════════════════════════════════════════"
