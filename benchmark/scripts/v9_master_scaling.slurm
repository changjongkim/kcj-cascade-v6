#!/bin/bash
# benchmark/scripts/v9_master_scaling.slurm
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_master_scaling
#SBATCH -o benchmark/logs/v9_master_scaling_%j.out
#SBATCH -e benchmark/logs/v9_master_scaling_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEMS=("cascade" "lmcache" "lmcache-redis" "hdf5-indep" "hdf5-coll")
NODES_LIST=(1 2 4 8 16 32)
JOB_UID=${SLURM_JOB_ID}

echo "============================================================"
echo "🌟 V9 Master Scaling Suite Started (32 Nodes Allocated)"
echo "============================================================"

for SYS in "${SYSTEMS[@]}"; do
    echo "------------------------------------------------------------"
    echo "🚀 Starting System: $SYS"
    echo "------------------------------------------------------------"
    
    # 1. System Specific Setup
    if [[ "$SYS" == "lmcache-redis" ]]; then
        echo "Starting Redis servers on all nodes..."
        srun --overlap -N 32 -n 32 --ntasks-per-node=1 redis-server --port 16379 --bind 0.0.0.0 --protected-mode no --save "" &
        sleep 15
    elif [[ "$SYS" == "hdf5-coll" ]]; then
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
    fi

    # 2. Scaling Loop within the same Job
    for N in "${NODES_LIST[@]}"; do
        echo "   [Scale: $N Nodes]"
        REQS=$((8 * N))
        RUN_ID="${JOB_UID}_${SYS}_N${N}"
        
        USE_MPI_IO=""
        if [[ "$SYS" == "hdf5-coll" ]]; then
            USE_MPI_IO="--use-mpi-io"
        fi

        # Run benchmark on N nodes
        srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
            --system $SYS \
            --num-requests $REQS \
            --run-id $RUN_ID \
            $USE_MPI_IO
        
        sleep 2
    done

    # 3. System Specific Cleanup
    if [[ "$SYS" == "lmcache-redis" ]]; then
        echo "Cleaning up Redis..."
        pkill -u $USER redis-server
        sleep 5
    elif [[ "$SYS" == "hdf5-coll" ]]; then
        unset LD_PRELOAD
        module unload cray-hdf5-parallel
    fi
    
    echo "✅ Finished System: $SYS"
    echo ""
done

echo "============================================================"
echo "🏁 V9 Master Scaling Suite Complete."
echo "============================================================"
