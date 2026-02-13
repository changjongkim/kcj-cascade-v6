#!/bin/bash
# build_cpp.sh - Fix for GLIBCXX and User settings

set -e

# Source the environment
source setup_env.sh

BUILD_DIR="cascade_Code/cpp/build_cascade_cpp"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Configuring with CMake..."
# Explicitly use the GCC 11.2 compilers and fix LD_LIBRARY_PATH for cmake
LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON=ON \
    -DUSE_MPI=ON \
    -DCMAKE_C_COMPILER=$(which gcc) \
    -DCMAKE_CXX_COMPILER=$(which g++) \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DPERLMUTTER=ON

echo "Building..."
# User specifically mentioned -j 258
# We use a fallback if it fails or gets killed
make -j 258 || make -j 32

echo "Build complete!"
