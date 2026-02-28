#!/bin/bash
# benchmark/scripts/v9_hdf5_coll_retry_4n.slurm
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_HDF5_Coll_Retry
#SBATCH -o benchmark/logs/v9_HDF5_Coll_Retry_4n_%j.out
#SBATCH -e benchmark/logs/v9_HDF5_Coll_Retry_4n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# 1. Standard NERSC Collective HDF5 setup
module load cray-hdf5-parallel
module load libfabric

# 2. Critical MPI stabilization for CXI fabric
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_DEFAULT_VNI=0
export FI_PROVIDER=cxi

# 3. Handle specific Perlmuttter fabric discovery
export MPICH_OFI_NIC_VERBOSE=1

# 4. Critical: Force system libfabric (Discover the link automatically)
SYSLIBFABRIC=$(ldd $(which fi_info) | grep libfabric | awk '{print $3}' | xargs dirname)
export LD_LIBRARY_PATH=$SYSLIBFABRIC:$LD_LIBRARY_PATH
export LD_PRELOAD=$SYSLIBFABRIC/libfabric.so.1

# 5. Flush some cache or sync
sync

# 6. Run with smaller number of requests to debug
srun --overlap -n 4 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system hdf5-coll \
    --num-requests 32 \
    --gen-tokens 32 \
    --use-mpi-io
