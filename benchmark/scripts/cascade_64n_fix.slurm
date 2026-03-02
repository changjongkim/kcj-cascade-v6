#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J cascade_64n_fix
#SBATCH -o benchmark/logs/cascade_64n_fix_%j.out
#SBATCH -e benchmark/logs/cascade_64n_fix_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Slingshot Fixes
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1

SYSTEM="cascade"
REQS=128

echo "🚀 Running Fixed Cascade Strong Scaling at 64 Nodes..."
srun --overlap -N 64 -n 64 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system $SYSTEM --num-requests $REQS --run-id "cascade_64n_fix_${SLURM_JOB_ID}"

echo "✅ Done"
