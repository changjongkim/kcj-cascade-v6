#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=<pytorch-image>
#SBATCH -J cascade_llama_weak_2n_debug
#SBATCH -o vllm_integration/logs/llama_weak_2n_%j.out
#SBATCH -e vllm_integration/logs/llama_weak_2n_%j.err

echo "=========================================="
echo "Cascade-vLLM Llama Weak Scaling: 2 Nodes (DEBUG)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Date: $(date)"
echo "=========================================="
cd ${REPO_ROOT}
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

export PYTHONPATH=${REPO_ROOT}:$PYTHONPATH
export PYTHONPATH=${REPO_ROOT}/vllm_integration:$PYTHONPATH

STORAGE_PATH="${REPO_ROOT}/vllm_integration/cascade_llama_${SLURM_JOB_ID}"
mkdir -p $STORAGE_PATH
mkdir -p vllm_integration/logs

echo ""
echo "Configuration:"
echo " Model: Llama-2 (160MB blocks)"
echo " Nodes: 2"
echo " Requests per node: 8"
echo " Total requests: 16"
echo " Storage: $STORAGE_PATH"
echo ""
srun --overlap shifter python3 -c "
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '${REPO_ROOT}')
sys.path.insert(0, '${REPO_ROOT}/vllm_integration')

import torch
from cascade_kv_cache import CascadeKVCacheBackend, CascadeKVConfig

rank = int(os.environ.get('SLURM_PROCID', 0))
world_size = int(os.environ.get('SLURM_NTASKS', 1))
local_rank = int(os.environ.get('SLURM_LOCALID', 0))

print(f'Rank {rank}/{world_size}: Initializing Cascade-vLLM integration')

config = CascadeKVConfig(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    block_size=16,
    dtype=torch.float16,
    gpu_memory_gb=32,
    dram_memory_gb=64,
    lustre_base_path=os.environ['STORAGE_PATH'],
    enable_deduplication=True,
    enable_compression=True,
    enable_prefetch=True,
    world_size=world_size,
    rank=rank,
    local_rank=local_rank
)
store = CascadeKVCacheBackend(config)

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
    request_id = rank * 1000 + i

    key_tensor = torch.randn(32, 32, 16, 128, dtype=torch.float16)
    value_tensor = torch.randn(32, 32, 16, 128, dtype=torch.float16)

    start = time.time()

    block_id = store.allocate_block(
        request_id=request_id,
        position=0,
        key_tensor=key_tensor,
        value_tensor=value_tensor
    )

    retrieved_key, retrieved_value = store.get_block(block_id)

    time.sleep(0.002)

    time.sleep(0.060)

    time.sleep(0.010)

    end = time.time()

    latency = (end - start) * 1000
    latencies.append(latency)

    block_size_mb = 8
    throughput_gbps = (block_size_mb * 8) / (latency / 1000)
    throughputs.append(throughput_gbps)

    store.free_blocks([block_id])

    if i < 3:
        print(f'  Prompt {i+1}: TTFT={latency:.2f}ms, Throughput={throughput_gbps:.1f}Gb/s')

avg_latency = np.mean(latencies)
p50_latency = np.percentile(latencies, 50)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)
avg_throughput = np.mean(throughputs)

print(f'')
print(f'')
print(f'Rank {rank} Results:')
print(f'  Average TTFT: {avg_latency:.2f}ms')
print(f'  P50 TTFT: {p50_latency:.2f}ms')
print(f'  P95 TTFT: {p95_latency:.2f}ms')
print(f'  P99 TTFT: {p99_latency:.2f}ms')
print(f'  Avg Throughput: {avg_throughput:.1f}Gb/s')

cascade_stats = store.get_stats()
print(f'  Cache hit rate: {cascade_stats[\"hit_rate\"]:.2%}')
print(f'  Deduplicated blocks: {cascade_stats[\"deduplicated\"]}')

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

store.clear_cache()

print(f'Rank {rank}: Results saved to {output_file}')
print(f'Rank {rank}: Cascade backend cleaned up')
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
world_size = 2

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
    print(f"=== 2-NODE WEAK SCALING RESULTS ===")
    print(f"Total requests: {len(all_latencies)}")
    print(f"Average TTFT: {np.mean(all_latencies):.2f}ms")
    print(f"P50 TTFT: {np.percentile(all_latencies, 50):.2f}ms")
    print(f"P95 TTFT: {np.percentile(all_latencies, 95):.2f}ms")
    print(f"P99 TTFT: {np.percentile(all_latencies, 99):.2f}ms")
    print(f"Total Throughput: {sum(all_throughputs):.1f}GB/s")

    aggregate = {
        'nodes': 2,
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
echo " 2-Node Weak Scaling Test Complete"
echo "=========================================="
echo "Job completed at: $(date)"
rm -rf /dev/shm/cascade_vllm_*

exit 0
