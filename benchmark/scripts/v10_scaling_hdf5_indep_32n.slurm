#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v10_hdf5_indep_32n
#SBATCH -o benchmark/logs/v10_hdf5_indep_32n_%j.out
#SBATCH -e benchmark/logs/v10_hdf5_indep_32n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh



SYSTEM="hdf5-indep"
NODES_LIST=(1 2 4 8 16 32)
JOB_UID=${SLURM_JOB_ID}

echo "🚀 Starting hdf5_indep Scaling (1-32 nodes)"

for N in "${NODES_LIST[@]}"; do
    echo "   [Scale: $N Nodes]"
    REQS=$((8 * N))
    RUN_ID="${JOB_UID}_N${N}"
    
    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py         --system $SYSTEM         --num-requests $REQS         --run-id $RUN_ID
    
    sleep 5
done
echo "✅ Finished hdf5_indep"
