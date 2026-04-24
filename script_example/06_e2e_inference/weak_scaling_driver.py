
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    USE_MPI = True
except ImportError:
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    USE_MPI = False

job_id = os.environ.get('SLURM_JOB_ID', 'local')

def print_rank0(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)

def barrier():
    if USE_MPI:
        comm.Barrier()
    else:

        bar_dir = Path("/tmp") / f"cascade_barrier_{job_id}"
        bar_dir.mkdir(parents=True, exist_ok=True)
        (bar_dir / f"rank_{rank}").touch()

        while len(list(bar_dir.iterdir())) < world_size:
            time.sleep(0.5)
        time.sleep(1)

class SyntheticKVLoader:
    def __init__(self, block_size_mb: int = 160, num_blocks: int = 1000):
        self.block_size_bytes = int(block_size_mb * 1024 * 1024)
        self.block_ids = [f"llama_block_{i}" for i in range(num_blocks)]
        self.num_blocks = num_blocks

        print_rank0(f"[SyntheticKVLoader] {num_blocks} blocks  {block_size_mb} MB = "
                   f"{num_blocks * block_size_mb / 1024:.2f} GB total")

    def load(self, block_id: str) -> Tuple[bytes, bytes]:
        seed = abs(hash(block_id)) % (2**31)
        rng = np.random.default_rng(seed)

        data = rng.integers(0, 256, self.block_size_bytes, dtype=np.uint8).tobytes()

        mid = len(data) // 2
        return data[:mid], data[mid:]

def benchmark_cascade_vllm(args):
    print_rank0("\n" + "="*60)
    print_rank0(f"Cascade-vLLM Weak Scaling Benchmark")
    print_rank0("="*60)
    print_rank0(f"Rank: {rank}/{world_size}")
    print_rank0(f"Requests: {args.num_requests}")
    print_rank0(f"Block Size: {args.block_size_mb} MB")
    print_rank0(f"Run ID: {args.run_id}")
    print_rank0("="*60)

    try:
        from cascade_block_allocator import CascadeBlockAllocator, CascadeConfig, StorageTier
        from cascade_attention_backend import CascadeAttentionBackend, CascadeAttentionConfig
        import torch
    except ImportError as e:
        print_rank0(f"Error importing Cascade modules: {e}")
        return None

    loader = SyntheticKVLoader(
        block_size_mb=args.block_size_mb,
        num_blocks=args.num_requests * 10
    )

    model_config = {
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'num_hidden_layers': 32,
        'vocab_size': 50257
    }

    parallel_config = {
        'tensor_parallel_size': 1,
        'pipeline_parallel_size': 1
    }

    cascade_config = CascadeConfig(
        gpu_capacity_gb=args.gpu_capacity_gb,
        shm_capacity_gb=args.shm_capacity_gb,
        lustre_path=args.storage_path,
        enable_dedup=args.enable_dedup,
        enable_compression=args.enable_compression,
        semantic_eviction=True,
        block_size=16
    )

    print_rank0("\n[1/4] Initializing Cascade BlockAllocator...")
    allocator = CascadeBlockAllocator(cascade_config, model_config, parallel_config)

    print_rank0(f"  GPU Blocks: {allocator.num_gpu_blocks}")
    print_rank0(f"  SHM Blocks: {allocator.num_shm_blocks}")
    print_rank0(f"  Block Size: {allocator.block_size_bytes / 1024 / 1024:.2f} MB")

    print_rank0("\n[2/4] Initializing Cascade AttentionBackend...")
    attention_config = CascadeAttentionConfig(
        enable_prefetch=args.enable_prefetch,
        prefetch_distance=4,
        num_cuda_streams=8,
        use_compression=args.enable_compression,
        compression_format="int8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = CascadeAttentionBackend(allocator, attention_config, device)

    print_rank0(f"  Device: {device}")
    print_rank0(f"  CUDA Streams: {len(backend.cuda_streams)}")

    barrier()

    print_rank0("\n[3/4] Running Warm-up Phase...")

    warmup_blocks = []
    for i in range(min(5, args.num_requests)):
        block_id = loader.block_ids[i]
        key_data, val_data = loader.load(block_id)

        logical_ids = allocator.allocate(1)
        if logical_ids:
            warmup_blocks.append(logical_ids[0])

    if warmup_blocks:
        key_cache, val_cache = backend.fetch_kv_blocks(warmup_blocks, layer_idx=0)
        print_rank0(f"  Warmed up {len(warmup_blocks)} blocks")

    barrier()

    print_rank0("\n[4/4] Running Main Benchmark...")

    latencies = []
    throughputs = []

    start_total = time.time()

    for req_idx in range(args.num_requests):

        blocks_per_request = 10
        request_blocks = []

        request_start = time.time()

        logical_ids = allocator.allocate(blocks_per_request)

        if req_idx == 0:
            backend.mark_prefix_blocks(logical_ids[:2])

        key_cache, val_cache = backend.fetch_kv_blocks(logical_ids, layer_idx=0)

        backend.update_block_access_time(logical_ids)

        request_end = time.time()
        request_latency = (request_end - request_start) * 1000

        latencies.append(request_latency)

        throughput = blocks_per_request / (request_end - request_start)
        throughputs.append(throughput)

        if rank == 0 and req_idx % 10 == 0:
            print(f"  Request {req_idx}/{args.num_requests}: {request_latency:.2f}ms, "
                  f"{throughput:.2f} blocks/s")

        request_blocks.extend(logical_ids)

        time.sleep(0.001)

    total_time = time.time() - start_total

    results = {
        'rank': rank,
        'world_size': world_size,
        'num_requests': args.num_requests,
        'block_size_mb': args.block_size_mb,
        'total_time': total_time,
        'avg_latency_ms': np.mean(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'avg_throughput': np.mean(throughputs),
        'total_throughput': np.sum(throughputs),
    }

    alloc_stats = allocator.get_stats()
    results.update({
        'gpu_blocks_used': alloc_stats['gpu_blocks'],
        'shm_blocks_used': alloc_stats['shm_blocks'],
        'dedup_ratio': alloc_stats['dedup_ratio'],
    })

    backend_stats = backend.get_performance_stats()
    results.update({
        'avg_transfer_time_ms': backend_stats['avg_transfer_time_ms'],
    })

    backend.shutdown()

    return results

def main():
    parser = argparse.ArgumentParser(description="Cascade-vLLM Weak Scaling Benchmark")

    parser.add_argument("--num-requests", type=int, default=16,
                       help="Number of requests to process")
    parser.add_argument("--block-size-mb", type=int, default=160,
                       help="Block size in MB (default: 160 for Llama)")
    parser.add_argument("--run-id", type=str, default=f"{job_id}",
                       help="Run identifier")

    parser.add_argument("--storage-path", type=str,
                       default="${REPO_ROOT}/vllm_integration/cascade_store_weak",
                       help="Lustre storage path")
    parser.add_argument("--gpu-capacity-gb", type=float, default=32.0,
                       help="GPU capacity in GB")
    parser.add_argument("--shm-capacity-gb", type=float, default=64.0,
                       help="SHM capacity in GB")

    parser.add_argument("--enable-dedup", action="store_true", default=True,
                       help="Enable deduplication")
    parser.add_argument("--enable-compression", action="store_true", default=True,
                       help="Enable compression")
    parser.add_argument("--enable-prefetch", action="store_true", default=True,
                       help="Enable prefetching")

    args = parser.parse_args()

    if rank == 0:
        Path(args.storage_path).mkdir(parents=True, exist_ok=True)

    barrier()

    results = benchmark_cascade_vllm(args)

    if results is None:
        print_rank0("Benchmark failed!")
        return

    all_results = None
    if USE_MPI:
        all_results = comm.gather(results, root=0)
    else:
        all_results = [results]

    if rank == 0:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        for r in all_results:
            print(f"\nRank {r['rank']}:")
            print(f"  Requests: {r['num_requests']}")
            print(f"  Total Time: {r['total_time']:.2f}s")
            print(f"  Avg Latency: {r['avg_latency_ms']:.2f}ms")
            print(f"  P95 Latency: {r['p95_latency_ms']:.2f}ms")
            print(f"  P99 Latency: {r['p99_latency_ms']:.2f}ms")
            print(f"  Avg Throughput: {r['avg_throughput']:.2f} blocks/s")
            print(f"  GPU Blocks Used: {r['gpu_blocks_used']}")
            print(f"  Dedup Ratio: {r['dedup_ratio']:.2f}x")

        total_requests = sum(r['num_requests'] for r in all_results)
        avg_latency = np.mean([r['avg_latency_ms'] for r in all_results])
        max_p99 = max(r['p99_latency_ms'] for r in all_results)
        total_throughput = sum(r['total_throughput'] for r in all_results)

        print("\n" + "="*60)
        print("AGGREGATE STATISTICS")
        print("="*60)
        print(f"Nodes: {world_size}")
        print(f"Total Requests: {total_requests}")
        print(f"Avg Latency: {avg_latency:.2f}ms")
        print(f"Max P99 Latency: {max_p99:.2f}ms")
        print(f"Total Throughput: {total_throughput:.2f} blocks/s")
        print(f"Block Size: {args.block_size_mb} MB")
        print("="*60)

        output_file = Path(args.storage_path).parent / f"results_weak_{args.run_id}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'individual': all_results,
                'aggregate': {
                    'nodes': world_size,
                    'total_requests': total_requests,
                    'avg_latency_ms': avg_latency,
                    'max_p99_latency_ms': max_p99,
                    'total_throughput': total_throughput,
                    'block_size_mb': args.block_size_mb,
                }
            }, f, indent=2)

        print(f"\n Results saved to: {output_file}")

if __name__ == "__main__":
    main()
