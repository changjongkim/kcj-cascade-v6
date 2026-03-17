#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v20_16n_cascade
#SBATCH -o benchmark/logs/v20_var_16n_cascade_%j.out
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "=== 🏁 Variable Block 16-Node Only: cascade ==="

if [ "cascade" == "redis" ]; then
    mkdir -p benchmark/tmp/hosts_$SLURM_JOB_ID
    hostname > benchmark/tmp/hosts_$SLURM_JOB_ID/redis_host
    /global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/redis-server --port 16379 --protected-mode no --save "" --appendonly no --maxmemory 100gb --maxmemory-policy allkeys-lru &
    REDIS_PID=$!
    sleep 30
fi

srun -N 16 -n 16 python3 benchmark/scripts/v20_variable_block_bench.py \
    --system cascade \
    --block-size-mb 160 --num-write-blocks 50 --num-read-ops 200 --sigma 0.8 \
    --redis-port 16379

if [ "cascade" == "redis" ]; then
    kill $REDIS_PID 2>/dev/null || true
    rm -rf benchmark/tmp/hosts_$SLURM_JOB_ID
fi
