#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v20_var_64n_redis_dist
#SBATCH -o benchmark/logs/v20_var_64n_redis-dist_%j.out
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

SYSTEM="redis-dist"
PORT=16379

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "=== 🏁 Variable Block 64-Node Scale: ${SYSTEM} ==="

JOB_TMP_DIR=benchmark/tmp/hosts_$SLURM_JOB_ID
mkdir -p $JOB_TMP_DIR
scontrol show hostnames $SLURM_JOB_NODELIST > $JOB_TMP_DIR/all_hosts

# Start redis-server on EACH node
srun --overlap -N 64 -n 64 --ntasks-per-node=1 \
    /global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/redis-server --port $PORT \
    --maxmemory 32gb --maxmemory-policy allkeys-lru \
    --protected-mode no --save "" --appendonly no &
REDIS_SRUN_PID=$!
sleep 30

# Run the variable block benchmark
srun -N 64 -n 64 python3 benchmark/scripts/v20_variable_block_bench.py \
    --system ${SYSTEM} \
    --block-size-mb 160 \
    --num-write-blocks 30 \
    --num-read-ops 100 \
    --sigma 0.8 \
    --redis-port $PORT

# Cleanup
pkill -u $USER redis-server
wait $REDIS_SRUN_PID 2>/dev/null
rm -rf $JOB_TMP_DIR

echo "✅ ${SYSTEM} 64-Node Variable Block experiment completed."
