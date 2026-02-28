#!/bin/bash
# benchmark/scripts/v9_scaling_hdf5_coll.slurm
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_scaling_hdf5_coll
#SBATCH -o benchmark/logs/v9_scaling_hdf5_coll_%j.out
#SBATCH -e benchmark/logs/v9_scaling_hdf5_coll_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Collective IO Env (Optimized for Perlmutter)
module load cray-hdf5-parallel
module load libfabric
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_DEFAULT_VNI=0
export FI_PROVIDER=cxi
SYSLIBFABRIC=/opt/cray/libfabric/1.22.0/lib64
export LD_LIBRARY_PATH=$SYSLIBFABRIC:$LD_LIBRARY_PATH
export LD_PRELOAD=$SYSLIBFABRIC/libfabric.so.1
export MPICH_MPIIO_HINTS="*:direct_io=1:romio_ds_write=disable"

SYSTEM="hdf5-coll"
NODES_LIST=(1 2 4 8 16 32)

for N in "${NODES_LIST[@]}"; do
    echo "============================================================"
    echo "🚀 Running $SYSTEM on $N nodes"
    echo "============================================================"
    REQS=$((8 * N))
    RUN_ID="${SLURM_JOB_ID}_N${N}"
    
    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID \
        --use-mpi-io
    
    echo "✅ Finished $N nodes"
    sleep 5
done
