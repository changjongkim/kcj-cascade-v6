#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -o vllm_integration/logs/cascade_e2e_%j.out
#SBATCH -e vllm_integration/logs/cascade_e2e_%j.err
#SBATCH -J cascade_e2e

echo "=========================================="
echo "Cascade+vLLM E2E Integration Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Date: $(date)"
echo "=========================================="

REPO_ROOT="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
CONDA_ENV="$REPO_ROOT/conda_env"
export PATH="$CONDA_ENV/bin:$PATH"

module load cudatoolkit/12.2

export HF_HOME=/pscratch/sd/s/sgkim/kcj/hf_cache

echo ""
echo "Environment:"
echo "  Python: $(which python3)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__} CUDA={torch.cuda.is_available()}')"

python3 -c "import vllm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Installing vLLM..."
    pip install vllm 2>&1 | tail -3
    echo ""
fi
python3 -c "import vllm; print(f'  vLLM: {vllm.__version__}')" 2>/dev/null || {
    echo "ERROR: vLLM installation failed"
    exit 1
}

export PYTHONPATH="$REPO_ROOT/vllm_integration:$PYTHONPATH"

MODEL="${MODEL:-facebook/opt-125m}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-64}"
STORAGE_PATH="$REPO_ROOT/vllm_integration/cascade_store_$SLURM_JOB_ID"
OUTPUT_DIR="$REPO_ROOT/vllm_integration/cascade_results_$SLURM_JOB_ID"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Prompts: $NUM_PROMPTS"
echo "  Max tokens: $MAX_TOKENS"
echo "  Storage: $STORAGE_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""

mkdir -p "$STORAGE_PATH" "$OUTPUT_DIR" "$REPO_ROOT/vllm_integration/logs"

echo "Running Cascade+vLLM E2E test..."
srun python3 "$REPO_ROOT/script_example/06_e2e_inference/cascade_e2e_driver.py" \
    --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --storage-path "$STORAGE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --skip-baseline

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS"
else
    echo "FAILED (exit code: $EXIT_CODE)"
fi
echo "Job completed at: $(date)"
echo "=========================================="
