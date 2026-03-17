#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v20_32n_lmcache
#SBATCH -o benchmark/logs/v20_var_32n_lmcache_%j.out
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "=== 🏁 Variable Block 32-Node Only: lmcache ==="

if [ "lmcache" == "redis" ]; then
    mkdir -p benchmark/tmp/hosts_$SLURM_JOB_ID
    hostname > benchmark/tmp/hosts_$SLURM_JOB_ID/redis_host
    /global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/redis-server --port 16379 --protected-mode no --save "" --appendonly no --maxmemory 100gb --maxmemory-policy allkeys-lru &
    REDIS_PID=$!
    sleep 30
fi

srun -N 32 -n 32 python3 benchmark/scripts/v20_variable_block_bench.py \
    --system lmcache \
    --block-size-mb 160 --num-write-blocks 50 --num-read-ops 200 --sigma 0.8 \
    --redis-port 16379

if [ "lmcache" == "redis" ]; then
    kill $REDIS_PID 2>/dev/null || true
    rm -rf benchmark/tmp/hosts_$SLURM_JOB_ID
fi
