#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v16_tail_cdf_${n}n
#SBATCH -o benchmark/logs/v16_tail_cdf_${n}n_%j.out
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# RDMA environment variables for Cascade
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20
export FI_CXI_RETRY_COUNT=100
export FI_CXI_RETRY_WAIT=200
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_OFLOW_BUF_COUNT=7
export FI_CXI_REQ_BUF_SIZE=16777216
export FI_MR_CACHE_MONITOR=disabled
export MPICH_OFI_NUM_NICS=1

SYSTEMS="cascade pdc lmcache llm-gpu hdf5"
BLOCK_SIZES="160 320"

echo "=== Tail Latency CDF Data Collection (16 Nodes) ==="

for sys in $SYSTEMS; do
    if [ "$sys" == "redis" ]; then
        echo ">>> Starting Redis Server..."
        mkdir -p benchmark/tmp/hosts_$SLURM_JOB_ID
        hostname > benchmark/tmp/hosts_$SLURM_JOB_ID/redis_host
        redis-server --port 16379 --protected-mode no --save "" --appendonly no &
        REDIS_PID=$!
        sleep 5
    fi

    for bs in $BLOCK_SIZES; do
        model_name="Llama"
        if [ "$bs" -eq "320" ]; then model_name="Qwen"; fi
        echo "▶ Testing $sys with $model_name ($bs MB blocks)"
        srun -N 16 -n 16 python3 benchmark/scripts/v12_tail_latency_bench.py \
            --system $sys --block-size-mb $bs --num-write-blocks 30 --num-read-ops 300
        sleep 5
    done

    if [ "$sys" == "redis" ]; then
        kill $REDIS_PID
        rm -rf benchmark/tmp/hosts_$SLURM_JOB_ID
    fi
done
