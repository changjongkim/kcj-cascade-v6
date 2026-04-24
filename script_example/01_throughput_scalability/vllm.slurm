#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v9_llm_gpu_32n
#SBATCH -o benchmark/logs/v9_llm_gpu_32n_%j.out
#SBATCH -e benchmark/logs/v9_llm_gpu_32n_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

SYSTEM="llm-gpu"
NODES_LIST=(1 2 4)
JOB_UID=${SLURM_JOB_ID}

echo "🚀 Starting LLM-GPU (vLLM-GPU Baseline) Scaling (1-32 nodes)"

for N in "${NODES_LIST[@]}"; do
    echo "   [Scale: $N Nodes]"
    REQS=$((8 * N))
    RUN_ID="${JOB_UID}_N${N}"

    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 script_example/01_throughput_scalability/throughput_driver.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID

    sleep 5
done
echo "✅ Finished LLM-GPU"
