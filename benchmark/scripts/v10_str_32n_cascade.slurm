#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v10_str_32n_cascade
#SBATCH -o benchmark/logs/v10_str_32n_cascade_%j.out
#SBATCH -e benchmark/logs/v10_str_32n_cascade_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software\nexport MPICH_OFI_STARTUP_CONNECT=1\nexport FI_CXI_DEFAULT_CQ_SIZE=131072\nexport FI_CXI_OFLOW_BUF_SIZE=8388608

SYSTEM="cascade"
NODES_LIST=(16 32)
JOB_UID=${SLURM_JOB_ID}

echo "🚀 Starting cascade Strong Scaling (16-32 nodes, 128 Reqs Total)"



for N in "${NODES_LIST[@]}"; do
    echo "   [Scale: $N Nodes]"
    REQS=128
    RUN_ID="${JOB_UID}_N${N}"
    
    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID
    
    sleep 5
done



echo "✅ Finished cascade"
