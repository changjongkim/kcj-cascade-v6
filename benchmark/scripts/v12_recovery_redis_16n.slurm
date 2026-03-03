#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J recovery_redis_16n
#SBATCH -o benchmark/logs/recovery_redis_16n_%j.out
#SBATCH -e benchmark/logs/recovery_redis_16n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# OFI Settings
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=32768
export FI_CXI_OFLOW_BUF_SIZE=2097152
export FI_CXI_DEFAULT_VNI=0

SYSTEM="lmcache-redis"
BLOCK_MB=160
export REDIS_PORT=16379

echo "=== Recovery Profiling: ${SYSTEM} ==="

mkdir -p benchmark/tmp/hosts_${SLURM_JOB_ID}
rm -f benchmark/tmp/hosts_${SLURM_JOB_ID}/redis_host
redis-server --port $REDIS_PORT --bind 0.0.0.0 --protected-mode no --save "" &
REDIS_PID=$!
sleep 15
echo $(hostname) > benchmark/tmp/hosts_${SLURM_JOB_ID}/redis_host

for i in 16; do
    echo ""
    echo "▶ SCALE: N=${i} nodes"
    srun --overlap -N $i -n $i --ntasks-per-node=1 \
        python3 benchmark/scripts/v12_recovery_profiling.py \
        --system $SYSTEM \
        --block-size-mb $BLOCK_MB \
        --num-blocks 10
    sleep 5
done

kill $REDIS_PID
echo "✅ Done"
