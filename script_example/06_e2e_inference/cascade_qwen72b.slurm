#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -o vllm_integration/logs/cascade_qwen72b_%N_%j.out
#SBATCH -e vllm_integration/logs/cascade_qwen72b_%N_%j.err

NODES=$SLURM_NNODES
JOB_ID=$SLURM_JOB_ID

echo "=========================================="
echo "Cascade+vLLM Strong Scaling (Qwen2.5-72B, TP=4): ${NODES}N"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo "Nodes: $NODES"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs per node: 4 (tensor_parallel_size=4)"
echo "Date: $(date)"
echo "=========================================="

REPO_ROOT="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
export PATH="$REPO_ROOT/conda_env/bin:$PATH"
export HF_HOME=/pscratch/sd/s/sgkim/kcj/hf_cache
export PYTHONPATH="$REPO_ROOT/vllm_integration:$REPO_ROOT/ml_workspace/build_ml_311:$PYTHONPATH"
export LD_LIBRARY_PATH="$REPO_ROOT/conda_env/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

module load cudatoolkit/12.9
module load cray-mpich

python3 -c "import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()} GPUs={torch.cuda.device_count()}')"
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"

MODEL="Qwen/Qwen2.5-72B"
STORAGE_PATH="$REPO_ROOT/vllm_integration/cascade_qwen72b_store_${NODES}N_${JOB_ID}"
OUTPUT_DIR="$REPO_ROOT/vllm_integration/cascade_qwen72b_${NODES}N_${JOB_ID}"

echo ""
echo "Model: $MODEL"
echo "Total requests: 128 (strong scaling)"
echo "Requests per node: $((128 / NODES))"
echo "Tensor parallel: 4 GPUs per node"
echo "Storage: $STORAGE_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$STORAGE_PATH" "$OUTPUT_DIR" "$REPO_ROOT/vllm_integration/logs"

srun python3 "$REPO_ROOT/vllm_integration/run_cascade_strong.py" \
    --model "$MODEL" \
    --max-tokens 64 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --target-tokens 512 \
    --max-model-len 576 \
    --max-num-seqs 16 \
    --enforce-eager \
    --storage-path "$STORAGE_PATH" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: ${NODES}N strong scaling (Qwen2.5-72B TP=4)"
    if [ -f "$OUTPUT_DIR/aggregate.json" ]; then
        echo ""
        python3 -c "
import json
with open('$OUTPUT_DIR/aggregate.json') as f:
    d = json.load(f)
print(f'Pass1 wall: {d[\"pass1\"][\"wall_time_s\"]:.3f}s  throughput: {d[\"pass1\"][\"aggregate_throughput_toks_s\"]:.1f} toks/s')
print(f'Pass2 wall: {d[\"pass2\"][\"wall_time_s\"]:.3f}s  throughput: {d[\"pass2\"][\"aggregate_throughput_toks_s\"]:.1f} toks/s')
print(f'Speedup: {d[\"speedup\"]:.2f}x')
"
    fi
else
    echo "FAILED (exit code: $EXIT_CODE)"
fi
echo "Job completed at: $(date)"
echo "=========================================="
