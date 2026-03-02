#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v11_high_dens_4n
#SBATCH -o benchmark/logs/v11_high_dens_4n_%j.out
#SBATCH -e benchmark/logs/v11_high_dens_4n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SCRIPT=benchmark/scripts/v9_serv_metrics_standalone.py
PYTHON_BIN=$(which python3)
SYSTEMS=("Cascade" "LMCache" "PDC" "LLM-GPU" "HDF5-Indep" "Redis")
DENSITY=128
N=4
TOTAL_REQS=$((N * DENSITY))

for SYS in "${SYSTEMS[@]}"; do
    if [ "$SYS" == "Redis" ]; then
        REDIS_HOST=$(hostname); REDIS_PORT=16379
        srun -N 1 -n 1 redis-cli -h $REDIS_HOST -p $REDIS_PORT shutdown 2>/dev/null || true; sleep 2
        srun -N 1 -n 1 --overlap redis-server --port $REDIS_PORT --protected-mode no --save "" --appendonly no & sleep 5
        export REDIS_HOST=$REDIS_HOST; export REDIS_PORT=$REDIS_PORT
    fi
    srun -N $N -n $N --gpus-per-node=4 $PYTHON_BIN $SCRIPT --system $SYS --num-requests $TOTAL_REQS --run-id "high_dens_n4_${SYS}"
    [ "$SYS" == "Redis" ] && srun -N 1 -n 1 redis-cli -h $REDIS_HOST -p $REDIS_PORT shutdown 2>/dev/null || true
done
