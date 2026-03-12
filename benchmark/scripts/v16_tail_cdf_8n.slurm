#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v16_tail_cdf_8n
#SBATCH -o benchmark/logs/v16_tail_cdf_8n_%j.out

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# Systems to compare
SYSTEMS="cascade pdc lmcache"
# Llama (160MB) and Qwen (320MB)
BLOCK_SIZES="160 320"

echo "=== Tail Latency CDF Data Collection (8 Nodes) ==="

for sys in $SYSTEMS; do
    for bs in $BLOCK_SIZES; do
        model_name="Llama"
        if [ "$bs" -eq "320" ]; then model_name="Qwen"; fi
        
        echo "▶ Testing $sys with $model_name ($bs MB blocks)"
        # Note: -n 8 ensures 1 task per node across all 8 nodes
        srun -N 8 -n 8 python3 benchmark/scripts/v12_tail_latency_bench.py \
            --system $sys \
            --block-size-mb $bs \
            --num-write-blocks 50 \
            --num-read-ops 500
        sleep 5
    done
done

echo "✅ All CDF data collected in benchmark/tmp/tail_res_<JOBID>"
