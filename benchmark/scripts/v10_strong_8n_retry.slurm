#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:15:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v10_strong_8n_retry
#SBATCH -o benchmark/logs/v10_strong_8n_retry_%j.out
#SBATCH -e benchmark/logs/v10_strong_8n_retry_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

JOB_UID=${SLURM_JOB_ID}
NODES_LIST=(1 2 4 8)

echo "🚀 Starting Cascade 8-Node Strong Scaling Retry"
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OFLOW_BUF_SIZE=8388608

# Retry specifically for 8 nodes to fill the gap
N=8
REQS=128
RUN_ID="${JOB_UID}_cascade_N${N}"
echo "   [Scale: $N Nodes]"
srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system cascade \
    --num-requests $REQS \
    --run-id $RUN_ID
sleep 5

echo "🚀 Starting HDF5-Indep 1,2,4,8-Node Strong Scaling Retry"
export HDF5_USE_FILE_LOCKING=FALSE

for N in "${NODES_LIST[@]}"; do
    echo "   [Scale: $N Nodes]"
    RUN_ID="${JOB_UID}_hdf5_indep_N${N}"
    
    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system hdf5-indep \
        --num-requests $REQS \
        --run-id $RUN_ID
    
    sleep 5
done

echo "✅ Finished 8n Retry"
