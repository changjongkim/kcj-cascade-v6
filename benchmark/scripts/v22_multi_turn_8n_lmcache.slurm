#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v22_mt_lmcache
#SBATCH -o benchmark/logs/v22_multi_turn_8n_lmcache_%j.out
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "=== 💬 Multi-Turn Scaling (1, 2, 4, 8 Nodes): lmcache ==="

for NODES in 1 2 4 8; do
    echo "--------------------------------------------------------"
    echo "▶ Scaling Step: $NODES Nodes"
    echo "--------------------------------------------------------"
    
    if [ "lmcache" == "redis" ]; then
        mkdir -p benchmark/tmp/hosts_$SLURM_JOB_ID_$NODES
        hostname > benchmark/tmp/hosts_$SLURM_JOB_ID_$NODES/redis_host
        /global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/redis-server --port 16379 --protected-mode no --save "" --appendonly no --maxmemory 100gb --maxmemory-policy allkeys-lru &
        REDIS_PID=$!
        sleep 30
    fi

    # Run Multi-Turn with scaling nodes
    srun -N $NODES -n $NODES python3 benchmark/scripts/v22_multi_turn_inc_bench.py \
        --system lmcache \
        --num-turns 10 \
        --block-size-mb 160 \
        --redis-port 16379

    if [ "lmcache" == "redis" ]; then
        kill $REDIS_PID 2>/dev/null || true
        rm -rf benchmark/tmp/hosts_$SLURM_JOB_ID_$NODES
    fi
    
    # Optional barrier or clear storage between steps if needed
    sleep 5
done

echo "✅ Multi-Turn Scaling for lmcache Completed."
