#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_dedup_redis
#SBATCH -o benchmark/logs/v12_dedup_redis_%j.out
#SBATCH -e benchmark/logs/v12_dedup_redis_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="lmcache-redis"
BLOCK_MB=160
PREFIX_BLOCKS=64
export REDIS_PORT=16379

echo "=== Evaluation 1: Global Deduplication & Prefix Sharing ==="
echo "System: ${SYSTEM}"

# Start Redis on the head node
mkdir -p benchmark/tmp/hosts_${SLURM_JOB_ID}
rm -f benchmark/tmp/hosts_${SLURM_JOB_ID}/redis_host

# Direct execution on the first node of the job
redis-server --port $REDIS_PORT --bind 0.0.0.0 --protected-mode no --save "" &
REDIS_PID=$!
sleep 15
echo $(hostname) > benchmark/tmp/hosts_${SLURM_JOB_ID}/redis_host

for i in 1 2 4 8; do
    echo ""
    echo "▶ SCALE: N=${i} nodes"
    srun --overlap -N $i -n $i --ntasks-per-node=1 \
        python3 benchmark/scripts/v12_dedup_prefix_bench.py \
        --system $SYSTEM \
        --block-size-mb $BLOCK_MB \
        --prefix-blocks $PREFIX_BLOCKS \
        --run-id "${SLURM_JOB_ID}_DEDUP_N${i}"
    sleep 5
done

# Cleanup
kill $REDIS_PID
echo "✅ Done"
