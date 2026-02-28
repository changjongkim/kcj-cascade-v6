#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_HDF5_Coll_4n
#SBATCH -o benchmark/logs/v9_HDF5_Coll_4n_%j.out
#SBATCH -e benchmark/logs/v9_HDF5_Coll_4n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Need to load parallel HDF5 environment for collective I/O
module load cray-hdf5-parallel
module load libfabric

# Perlmutter MPI stabilization flags
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_DEFAULT_VNI=0
export FI_PROVIDER=cxi

# MPI-IO Lustre Optimization
export MPICH_MPIIO_HINTS="*:direct_io=1;romio_ds_write=disable"

# Force use of mpio driver for collective I/O
srun --overlap -n 4 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system hdf5-coll \
    --num-requests 32 \
    --gen-tokens 32 \
    --use-mpi-io
