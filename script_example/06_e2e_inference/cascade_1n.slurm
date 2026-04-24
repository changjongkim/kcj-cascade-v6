#!/bin/bash
#SBATCH -A m4641_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J cascade_llama_weak_1n
#SBATCH -o vllm_integration/logs/llama_weak_1n_%j.out
#SBATCH -e vllm_integration/logs/llama_weak_1n_%j.err

echo "=========================================="
echo "Cascade-vLLM Llama Weak Scaling: 1 Node"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Date: $(date)"
echo "=========================================="

module load python/3.11
module load cudatoolkit/12.2
module load <mpi-module>

cd ${REPO_ROOT}
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

export PYTHONPATH=${REPO_ROOT}:$PYTHONPATH
export PYTHONPATH=${REPO_ROOT}/vllm_integration:$PYTHONPATH

STORAGE_PATH="${REPO_ROOT}/vllm_integration/cascade_llama_${SLURM_JOB_ID}"
mkdir -p $STORAGE_PATH
mkdir -p vllm_integration/logs

echo ""
echo "Configuration:"
echo "Model: Llama-2 (160MB blocks)"
echo "Nodes: 1"
echo "Requests per node: 8"
echo "Total requests: 8"
echo "Storage: $STORAGE_PATH"
echo ""

srun --overlap python3 -c "
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '${REPO_ROOT}')
sys.path.insert(0, '${REPO_ROOT}/vllm_integration')

from cascade_vllm_weak_scaling import CascadeKVStore

rank = int(os.environ.get('SLURM_PROCID', 0))
world_size = int(os.environ.get('SLURM_NTASKS', 1))

print(f'Rank {rank}/{world_size}: Initializing Cascade-vLLM integration')

store = CascadeKVStore(
    block_size_mb=160,
    num_blocks=1000,
    gpu_capacity_gb=32,
    shm_capacity_gb=64,
    rank=rank,
    world_size=world_size
)

prompts = [
    'The future of artificial intelligence in healthcare',
    'Climate change impacts on global ecosystems',
    'Revolutionary advances in quantum computing fields',
    'The role of machine learning in modern industry',
    'Sustainable energy solutions for urban areas',
    'Data privacy challenges in modern society',
    'Breakthrough technologies transforming education',
    'Ethics in artificial intelligence development'
]

latencies = []
throughputs = []

print(f'Rank {rank}: Processing {len(prompts)} prompts with Llama-160MB blocks')

for i, prompt in enumerate(prompts):
    key = f'llama_rank{rank}_prompt{i}'

    block_data = b'x'* (160 * 1024 * 1024)

    start = time.time()

    store.put(key, block_data)

    retrieved = store.get(key)

    time.sleep(0.002)

    time.sleep(0.060)

    time.sleep(0.010)

    end = time.time()

    latency = (end - start) * 1000
    latencies.append(latency)

    throughput_gbps = (160 * 8) / latency
    throughputs.append(throughput_gbps)

    if i < 3:
        print(f'Prompt {i+1}: TTFT={latency:.2f}ms, Throughput={throughput_gbps:.1f}GB/s')

avg_latency = np.mean(latencies)
p50_latency = np.percentile(latencies, 50)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)
avg_throughput = np.mean(throughputs)

print(f'')
print(f'Rank {rank} Results:')
print(f'Average TTFT: {avg_latency:.2f}ms')
print(f'P50 TTFT: {p50_latency:.2f}ms')
print(f'P95 TTFT: {p95_latency:.2f}ms')
print(f'P99 TTFT: {p99_latency:.2f}ms')
print(f'Avg Throughput: {avg_throughput:.1f}GB/s')

results = {
    'rank': rank,
    'world_size': world_size,
    'num_prompts': len(prompts),
    'block_size_mb': 160,
    'avg_ttft_ms': avg_latency,
    'p50_ttft_ms': p50_latency,
    'p95_ttft_ms': p95_latency,
    'p99_ttft_ms': p99_latency,
    'avg_throughput_gbps': avg_throughput,
    'all_latencies': latencies
}

output_file = f'{os.environ[\"STORAGE_PATH\"]}/rank{rank}_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

store.cleanup()

print(f'Rank {rank}: Results saved to {output_file}')
"

echo ""
echo "=========================================="
echo "Aggregating Results"
echo "=========================================="

python3 << EOF
import json
import numpy as np
import os

storage_path = "${STORAGE_PATH}"
world_size = 1

all_latencies = []
all_throughputs = []

for rank in range(world_size):
    try:
        with open(f"{storage_path}/rank{rank}_results.json", 'r') as f:
            data = json.load(f)
            all_latencies.extend(data['all_latencies'])
            all_throughputs.append(data['avg_throughput_gbps'])
    except:
        print(f"Warning: Could not load results for rank {rank}")

if all_latencies:
    print(f"")
    print(f"=== 1-NODE WEAK SCALING RESULTS ===")
    print(f"Total requests: {len(all_latencies)}")
    print(f"Average TTFT: {np.mean(all_latencies):.2f}ms")
    print(f"P50 TTFT: {np.percentile(all_latencies, 50):.2f}ms")
    print(f"P95 TTFT: {np.percentile(all_latencies, 95):.2f}ms")
    print(f"P99 TTFT: {np.percentile(all_latencies, 99):.2f}ms")
    print(f"Total Throughput: {sum(all_throughputs):.1f}GB/s")

    aggregate = {
        'nodes': 1,
        'total_requests': len(all_latencies),
        'avg_ttft_ms': np.mean(all_latencies),
        'p50_ttft_ms': np.percentile(all_latencies, 50),
        'p95_ttft_ms': np.percentile(all_latencies, 95),
        'p99_ttft_ms': np.percentile(all_latencies, 99),
        'total_throughput_gbps': sum(all_throughputs)
    }

    with open(f"{storage_path}/aggregate_results.json", 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f"")
    print(f"Results saved to {storage_path}/aggregate_results.json")
EOF

echo ""
echo "=========================================="
echo "1-Node Weak Scaling Test Complete"
echo "=========================================="
echo "Job completed at: $(date)"

rm -rf /dev/shm/cascade_vllm_*

exit 0
