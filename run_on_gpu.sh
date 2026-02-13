#!/bin/bash
# run_on_gpu.sh

module load cudatoolkit
module load cray-mpich

# Set MPI GPU support
export MPICH_GPU_SUPPORT_ENABLED=1

# Add build directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/cascade_Code/cpp/build_cascade_cpp
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Perlmutter specific libfabric for MPI
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH

source setup_env.sh

# Run the benchmark
python3 bench_multi_node.py
