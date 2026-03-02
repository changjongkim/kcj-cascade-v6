#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J cascade_16n_fix
#SBATCH -o benchmark/logs/cascade_16n_fix_%j.out
#SBATCH -e benchmark/logs/cascade_16n_fix_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="cascade"
REQS=128

echo "🚀 Running Fixed Cascade Strong Scaling at 16 Nodes..."
srun --overlap -N 16 -n 16 --ntasks-per-node=1 python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system $SYSTEM --num-requests $REQS --run-id "cascade_16n_fix_${SLURM_JOB_ID}"

echo "✅ Done"
