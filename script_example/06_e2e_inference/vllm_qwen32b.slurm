#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -o vllm_integration/logs/baseline_qwen32b_%N_%j.out
#SBATCH -e vllm_integration/logs/baseline_qwen32b_%N_%j.err

NODES=$SLURM_NNODES
JOB_ID=$SLURM_JOB_ID

echo "=========================================="
echo "Pure vLLM Baseline (Qwen2.5-32B, TP=2): ${NODES}N"
echo "=========================================="
echo "Job ID: $JOB_ID  Nodes: $NODES  Date: $(date)"
echo "=========================================="

REPO_ROOT="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
export PATH="$REPO_ROOT/conda_env/bin:$PATH"
export HF_HOME=/pscratch/sd/s/sgkim/kcj/hf_cache
export LD_LIBRARY_PATH="$REPO_ROOT/conda_env/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

module load cudatoolkit/12.9

MODEL="Qwen/Qwen2.5-32B"
OUTPUT_DIR="$REPO_ROOT/vllm_integration/baseline_qwen32b_${NODES}N_${JOB_ID}"
mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/vllm_integration/logs"

srun python3 "$REPO_ROOT/vllm_integration/run_vllm_baseline.py" \
    --model "$MODEL" \
    --max-tokens 64 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --target-tokens 2048 \
    --max-model-len 2112 \
    --max-num-seqs 32 \
    --enforce-eager \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: ${NODES}N baseline (Qwen2.5-32B TP=2)"
else
    echo "FAILED (exit code: $EXIT_CODE)"
fi
echo "Job completed at: $(date)"
