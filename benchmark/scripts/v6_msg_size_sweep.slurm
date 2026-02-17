#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_msg_sweep
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/msg_sweep_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/msg_sweep_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

PYTHON_BIN=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/python3
SCRIPT=benchmark/scripts/v6_msg_size_sweep.py

echo "###################################################################"
echo " MESSAGE SIZE SWEEP: Cascade vs LMCache (Contention Mode)"
echo "###################################################################"

for N in 1 2 4 8; do
    echo ""
    echo ">>>> Testing with N=$N Nodes <<<<"
    srun -N $N -n $N --gpus-per-node=4 $PYTHON_BIN $SCRIPT
done
