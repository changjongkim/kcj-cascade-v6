#!/bin/bash
# build_cpp.sh - Fix for GLIBCXX and User settings

set -e

# Source the environment
source setup_env.sh

# Bypass conda's broken pkg-config
mkdir -p /tmp/sgkim_pkg
ln -sf /usr/bin/pkg-config /tmp/sgkim_pkg/pkg-config
export PATH=/tmp/sgkim_pkg:$PATH

BUILD_DIR="cascade_Code/cpp/build_cascade_cpp"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Configuring with CMake..."
# Use Cray compiler wrappers (cc, CC) which handle MPI and Cray libraries automatically
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON=ON \
    -DUSE_MPI=ON \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
    -DPKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config \
    -DPERLMUTTER=ON

echo "Building..."
# User specifically mentioned -j 258
# We use a fallback if it fails or gets killed
make -j 258 || make -j 32

echo "Build complete!"
