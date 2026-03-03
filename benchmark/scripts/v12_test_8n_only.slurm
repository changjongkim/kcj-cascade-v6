#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v12_test_8n_only
#SBATCH -o benchmark/logs/v12_test_8n_only_%j.out
#SBATCH -e benchmark/logs/v12_test_8n_only_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=10

# Add this just in case
export FI_CXI_DEFAULT_VNI=0

JOB_UID=${SLURM_JOB_ID}

echo "=== Llama 160MB | cascade | Strong 8N ONLY ==="
N=8
REQS=128

srun --overlap -N $N -n $N --ntasks-per-node=1 \
    python3 benchmark/scripts/v9_serv_metrics_standalone.py \
    --system cascade \
    --num-requests 128 \
    --block-size-mb 160 \
    --run-id "${JOB_UID}_LLAMA_STRONG_N8_ONLY"

echo "✅ Done"
