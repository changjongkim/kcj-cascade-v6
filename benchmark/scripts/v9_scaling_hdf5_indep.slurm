#!/bin/bash
# benchmark/scripts/v9_scaling_hdf5_indep.slurm
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_scaling_hdf5_indep
#SBATCH -o benchmark/logs/v9_scaling_hdf5_indep_%j.out
#SBATCH -e benchmark/logs/v9_scaling_hdf5_indep_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="hdf5-indep"
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
        --run-id $RUN_ID
    
    echo "✅ Finished $N nodes"
    sleep 5
done
