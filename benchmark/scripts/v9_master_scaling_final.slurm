#!/bin/bash
# benchmark/scripts/v9_master_scaling_final.slurm
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

# 🚀 [V9 Final Scaling Suite] 
# Systems to Compare:
# 1. Cascade (RDMA Optimized)
# 2. LMCache-Disk (Lustre File Store)  <- This is the 'lmcache-disk' version
# 3. LMCache-Redis (Remote In-Memory)
# 4. HDF5-Independent (File-per-rank)

SYSTEMS=("cascade" "lmcache-disk" "lmcache-redis" "hdf5-indep")
NODES_LIST=(1 2 4 8 16 32)
JOB_UID=${SLURM_JOB_ID}

echo "============================================================"
echo "🌟 V9 Master Scaling Suite Started (32 Nodes Allocated)"
echo "Metrics: Avg TTFT [ms], Total Throughput [req/s]"
echo "============================================================"

for SYS_NAME in "${SYSTEMS[@]}"; do
    echo "------------------------------------------------------------"
    echo "🚀 Testing System: $SYS_NAME"
    echo "------------------------------------------------------------"
    
    # Map name to internal adapter name
    INTERNAL_SYS=$SYS_NAME
    if [[ "$SYS_NAME" == "lmcache-disk" ]]; then INTERNAL_SYS="lmcache"; fi

    # 1. System Specific Setup
    if [[ "$SYS_NAME" == "lmcache-redis" ]]; then
        echo "Starting Redis servers on all allocated nodes..."
        srun --overlap -N 32 -n 32 --ntasks-per-node=1 redis-server --port 16379 --bind 0.0.0.0 --protected-mode no --save "" &
        sleep 20
    fi

    # 2. Sequential Scaling (1-32 nodes)
    for N in "${NODES_LIST[@]}"; do
        echo "   [Scale: $N Nodes]"
        REQS=$((8 * N))
        RUN_ID="${JOB_UID}_${SYS_NAME}_N${N}"
        
        # Run benchmark on subset of 32 nodes
        srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
            --system $INTERNAL_SYS \
            --num-requests $REQS \
            --run-id $RUN_ID
        
        sleep 10
    done

    # 3. Cleanup after each system test
    if [[ "$SYS_NAME" == "lmcache-redis" ]]; then
        echo "Stopping Redis servers..."
        pkill -u $USER redis-server
        sleep 5
    fi
    
    echo "✅ Finished System: $SYS_NAME"
    echo ""
done

echo "============================================================"
echo "🏁 V9 Master Scaling Suite Complete."
echo "============================================================"
