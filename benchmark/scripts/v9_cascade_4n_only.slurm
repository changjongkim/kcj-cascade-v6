#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_cascade_4n
#SBATCH -o benchmark/logs/v9_cascade_4n_%j.out
#SBATCH -e benchmark/logs/v9_cascade_4n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Workaround for OFI event counter exhaustion in Slingshot
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1

SYSTEM="cascade"
JOB_UID=${SLURM_JOB_ID}
N=4
REQS=$((8 * N))
RUN_ID="${JOB_UID}_N${N}"

echo "🚀 Starting Cascade 4-Node Run"
srun --overlap -N $N -n $N --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system $SYSTEM \
    --num-requests $REQS \
    --run-id $RUN_ID

echo "✅ Finished Cascade 4-Node"
