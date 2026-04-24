#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J cascade_vllm_e2e_2n
#SBATCH -o vllm_integration/logs/e2e_weak_2n_%j.out
#SBATCH -e vllm_integration/logs/e2e_weak_2n_%j.err

echo "=========================================="
echo "Cascade-vLLM End-to-End Weak Scaling"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Date: $(date)"
echo "=========================================="

module load python/3.11
module load cudatoolkit/12.2
module load cray-mpich

cd ${REPO_ROOT}

export PYTHONPATH=${REPO_ROOT}:$PYTHONPATH

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1

mkdir -p vllm_integration/logs
STORAGE_PATH="${REPO_ROOT}/vllm_integration/cascade_e2e_${SLURM_JOB_ID}"
mkdir -p $STORAGE_PATH

echo ""
echo "Environment:"
echo "  Python: $(which python3)"
echo "  Storage: $STORAGE_PATH"
echo ""

echo "=========================================="
echo "Step 1: Checking vLLM installation"
echo "=========================================="
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install --user vllm
    if [ $? -eq 0 ]; then
        echo "✓ vLLM installed successfully"
    else
        echo "✗ Failed to install vLLM - will use mock mode"
    fi
else
    echo "✓ vLLM already installed"
fi

echo ""
echo "Checking components:"
python3 -c "
import sys
sys.path.insert(0, 'vllm_integration')
try:
    import torch
    print(f'  ✓ PyTorch: {torch.__version__}')
    print(f'    CUDA available: {torch.cuda.is_available()}')
except:
    print('  ✗ PyTorch not available')

try:
    import vllm
    print(f'  ✓ vLLM: {vllm.__version__}')
except:
    print('  ✗ vLLM not available - will use mock mode')

try:
    from cascade_block_allocator import CascadeBlockAllocator
    print('  ✓ Cascade components available')
except Exception as e:
    print(f'  ✗ Cascade import failed: {e}')
"

echo ""
echo "=========================================="
echo "Step 2: Single Node Test (N=1)"
echo "=========================================="

PROMPTS=(
    "The future of artificial intelligence is"
    "Climate change is one of the most pressing issues because"
    "The key to successful software engineering is"
    "Machine learning models are becoming"
    "The importance of data privacy in modern society"
    "Quantum computing will revolutionize"
    "The role of ethics in technology development"
    "Sustainable energy solutions include"
)

echo "Running with 8 prompts on 1 node..."
srun --overlap -N 1 -n 1 \
  python3 script_example/06_e2e_inference/cascade_vllm_engine.py \
  --model "facebook/opt-125m" \
  --prompts "${PROMPTS[@]:0:8}" \
  --max-tokens 64 \
  > ${STORAGE_PATH}/n1_results.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✓ N=1 test completed"
    echo ""
    echo "Results preview:"
    grep -E "TTFT:|Throughput:|Avg TTFT" ${STORAGE_PATH}/n1_results.txt | head -10
else
    echo "✗ N=1 test failed"
    tail -20 ${STORAGE_PATH}/n1_results.txt
fi

echo ""
echo "Waiting 5 seconds before N=2 test..."
sleep 5

echo "=========================================="
echo "Step 3: Two Node Test (N=2)"
echo "=========================================="

cat > ${STORAGE_PATH}/run_distributed.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
sys.path.insert(0, 'vllm_integration')

from cascade_vllm_engine import run_llm_with_cascade

rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))

all_prompts = [
    "The future of artificial intelligence is",
    "Climate change is one of the most pressing issues because",
    "The key to successful software engineering is",
    "Machine learning models are becoming",
    "The importance of data privacy in modern society",
    "Quantum computing will revolutionize",
    "The role of ethics in technology development",
    "Sustainable energy solutions include",
    "Natural language processing has evolved",
    "Computer vision applications in healthcare",
    "Robotics and automation will change",
    "The impact of social media on society",
    "Cybersecurity threats in the digital age",
    "Blockchain technology promises to",
    "Virtual reality and the metaverse",
    "Edge computing enables new possibilities",
]

my_prompts = all_prompts[rank*8:(rank+1)*8]

print(f"\nRank {rank}/{world}: Processing {len(my_prompts)} prompts")

results = run_llm_with_cascade(
    model_name="facebook/opt-125m",
    prompts=my_prompts,
    max_tokens=64,
    use_cascade=True
)

if results:
    summary = results['summary']
    print(f"\nRank {rank} Results:")
    print(f"  Prompts: {summary['num_prompts']}")
    print(f"  Avg TTFT: {summary['avg_ttft_ms']:.2f}ms")
    print(f"  Total tokens: {summary.get('total_tokens', 0)}")

    import json
    output_file = f"cascade_e2e_{os.environ['SLURM_JOB_ID']}/rank{rank}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
EOF

echo "Running with 16 prompts on 2 nodes (8 per node)..."
srun --overlap -N 2 -n 2 \
  python3 ${STORAGE_PATH}/run_distributed.py \
  > ${STORAGE_PATH}/n2_results.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✓ N=2 test completed"
    echo ""
    echo "Results preview:"
    grep -E "Rank.*Results:|TTFT:|tokens:" ${STORAGE_PATH}/n2_results.txt | head -15
else
    echo "✗ N=2 test failed"
    tail -20 ${STORAGE_PATH}/n2_results.txt
fi

echo ""
echo "=========================================="
echo "Step 4: Results Analysis"
echo "=========================================="

python3 << EOF
import json
import sys
import os

storage_path = "${STORAGE_PATH}"

print("Attempting to load results...")

n1_ttft = None
try:
    with open(f"{storage_path}/n1_results.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Avg TTFT:" in line:
                parts = line.split("Avg TTFT:")[-1].split("ms")[0]
                n1_ttft = float(parts.strip())
                break
except Exception as e:
    print(f"Could not parse N=1 results: {e}")

n2_ttfts = []
try:
    for rank in range(2):
        json_file = f"{storage_path}/rank{rank}_results.json"
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                n2_ttfts.append(data['summary']['avg_ttft_ms'])
        else:
            with open(f"{storage_path}/n2_results.txt", 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if f"Rank {rank}" in line and "Avg TTFT:" in line:
                        parts = line.split("Avg TTFT:")[-1].split("ms")[0]
                        n2_ttfts.append(float(parts.strip()))
except Exception as e:
    print(f"Could not parse N=2 results: {e}")

print("\n" + "="*60)
print("WEAK SCALING ANALYSIS (End-to-End with LLM)")
print("="*60)

if n1_ttft:
    print(f"\n📊 N=1 (8 prompts):")
    print(f"  Avg TTFT: {n1_ttft:.2f}ms")
else:
    print("\n⚠️ N=1 results not available")

if n2_ttfts:
    avg_n2 = sum(n2_ttfts) / len(n2_ttfts)
    print(f"\n📊 N=2 (16 prompts, 8 per node):")
    print(f"  Rank 0 TTFT: {n2_ttfts[0]:.2f}ms")
    if len(n2_ttfts) > 1:
        print(f"  Rank 1 TTFT: {n2_ttfts[1]:.2f}ms")
    print(f"  Average TTFT: {avg_n2:.2f}ms")

    if n1_ttft:
        ratio = avg_n2 / n1_ttft
        print(f"\n📈 Weak Scaling:")
        print(f"  TTFT Ratio (N2/N1): {ratio:.3f}x")

        if ratio < 1.2:
            print(f"  ✓ EXCELLENT: TTFT increased only {(ratio-1)*100:.1f}%")
        elif ratio < 1.5:
            print(f"  ~ GOOD: TTFT increased {(ratio-1)*100:.1f}%")
        else:
            print(f"  ✗ POOR: TTFT increased {(ratio-1)*100:.1f}%")
else:
    print("\n⚠️ N=2 results not available")

print("\n" + "="*60)

print("\n📚 COMPARISON:")
print("Storage-only benchmark (from earlier):")
print("  N=1: 0.59ms")
print("  N=2: 0.62ms")
print("\nEnd-to-end with actual LLM:")
if n1_ttft:
    print(f"  N=1: {n1_ttft:.2f}ms ({n1_ttft/0.59:.0f}x slower)")
if n2_ttfts and len(n2_ttfts) > 0:
    print(f"  N=2: {sum(n2_ttfts)/len(n2_ttfts):.2f}ms ({sum(n2_ttfts)/len(n2_ttfts)/0.62:.0f}x slower)")

print("\nThe difference shows the actual cost of:")
print("  • Tokenization")
print("  • Prefill (attention computation)")
print("  • Token generation")
print("  • Model forward pass")
print("="*60)
EOF

echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Storage location: $STORAGE_PATH"
echo "Output files:"
ls -lh $STORAGE_PATH/*.txt 2>/dev/null
ls -lh $STORAGE_PATH/*.json 2>/dev/null

echo ""
echo "=========================================="
echo "✓ End-to-End Weak Scaling Test Complete!"
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

rm -rf /dev/shm/cascade_vllm_*

exit 0
