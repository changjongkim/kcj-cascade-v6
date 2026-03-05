import os
import time
import argparse
import numpy as np
from pathlib import Path
import json
import sys

# Add repo root to path for adapters
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

# Import adapter factory
from benchmark.run_benchmark import get_adapter

# MPI rank/world from SLURM
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

def print_rank0(*args):
    if rank == 0: 
        print(*args, flush=True)

class SyntheticKVLoader:
    def __init__(self, block_size_bytes: int):
        self.block_size = block_size_bytes
    
    def load(self, block_id: str):
        # Deterministic content based on block_id
        seed = abs(hash(block_id)) % (2**31)
        rng = np.random.default_rng(seed)
        data = rng.integers(0, 256, self.block_size, dtype=np.uint8).tobytes()
        mid = len(data) // 2
        return data[:mid], data[mid:]

def run_mixed_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--block-size-mb", type=int, default=16) # Smaller for intensive concurrent test
    parser.add_argument("--prefill-blocks", type=int, default=200)
    args = parser.parse_args()

    block_size_bytes = args.block_size_mb * 1024 * 1024
    loader = SyntheticKVLoader(block_size_bytes)
    system_name = args.system
    
    # Config setup
    config = {}
    if system_name.lower() == "cascade":
        config = {"gpu_capacity_gb": 36.0, "shm_capacity_gb": 128.0, "use_gpu": True, "use_compression": True}
    elif "redis" in system_name.lower():
        r_port = int(os.environ.get("REDIS_PORT", 16379))
        shared_redis_host = "localhost"
        tmp_h_dir = REPO_ROOT / "benchmark" / "tmp" / f"hosts_{job_id}"
        if tmp_h_dir.exists() and (tmp_h_dir / "redis_host").exists():
            with open(tmp_h_dir / "redis_host", 'r') as f:
                shared_redis_host = f.read().strip()
        config = {"host": shared_redis_host, "port": r_port}
    elif system_name.lower() == "lmcache":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/benchmark/lmcache_mixed_{job_id}"}
    elif system_name.lower() == "pdc":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/benchmark/pdc_mixed_{job_id}"}
    elif "hdf5" in system_name.lower():
        config = {"file_path": f"/pscratch/sd/s/sgkim/kcj/benchmark/tmp/h5_mixed_{job_id}.h5", "use_mpi": True}
    elif system_name.lower() == "llm-gpu":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/benchmark/llmgpu_mixed_{job_id}"}

    adapter = get_adapter(system_name, config)
    if not adapter.initialize():
        print(f"❌ [{system_name}] Failed to initialize on Rank {rank}")
        return

    # Workloads to test
    workloads = {
        "A": {"name": "Read-Heavy (95/5)", "read_prob": 0.95, "scan": False},
        "B": {"name": "Write-Heavy (50/50)", "read_prob": 0.50, "scan": False},
        "C": {"name": "Scan (100% Read)", "read_prob": 1.0, "scan": True}
    }

    final_results = {}

    for wl_id, wl in workloads.items():
        print_rank0(f"\n▶ Testing Workload {wl_id}: {wl['name']}")
        
        # 1. Initialization & Prefill
        adapter.clear()
        prefill_keys = []
        for i in range(args.prefill_blocks):
            key = f"prefill_rank{rank}_block{i}"
            k, v = loader.load(key)
            adapter.put(key, k, v)
            prefill_keys.append(key)
        adapter.flush()
        
        # Barrier (simple sleep if MPI barrier is risky)
        time.sleep(5)
        
        # 2. Benchmark Loop
        latencies = []
        ops_count = 0
        read_hits = 0
        errors = 0
        
        start_time = time.perf_counter()
        end_time = start_time + args.duration
        
        # For Scan workload, prepare sequential keys
        scan_idx = 0
        
        while time.perf_counter() < end_time:
            op_start = time.perf_counter()
            is_read = np.random.random() < wl['read_prob']
            
            if is_read:
                # Select a random key from ANY rank's prefill to test cross-node read if supported
                target_rank = np.random.randint(0, world)
                target_idx = np.random.randint(0, args.prefill_blocks)
                if wl['scan']:
                    target_idx = scan_idx % args.prefill_blocks
                    scan_idx += 1
                
                key = f"prefill_rank{target_rank}_block{target_idx}"
                res = adapter.get(key)
                if res: read_hits += 1
            else:
                # Write an update or a new unique block
                key = f"dynamic_rank{rank}_block{ops_count}"
                k, v = loader.load(key)
                adapter.put(key, k, v)
            
            latencies.append((time.perf_counter() - op_start) * 1000) # ms
            ops_count += 1
            
        actual_duration = time.perf_counter() - start_time
        
        # 3. Simple aggregation (Each rank saves its own stats to a temp file, Rank 0 reads them)
        stats = {
            "rank": rank,
            "ops": ops_count,
            "duration": actual_duration,
            "latencies": latencies, # This might be large, but for 60s it should be fine
            "hits": read_hits
        }
        
        summary_dir = REPO_ROOT / "benchmark" / "tmp" / f"agg_{job_id}_{wl_id}"
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / f"rank_{rank}.json", 'w') as f:
            json.dump(stats, f)
            
        # Wait for all
        time.sleep(10)
        
        if rank == 0:
            all_ops = 0
            all_lats = []
            max_dur = 0
            for r in range(world):
                try:
                    with open(summary_dir / f"rank_{r}.json", 'r') as f:
                        d = json.load(f)
                        all_ops += d['ops']
                        all_lats.extend(d['latencies'])
                        max_dur = max(max_dur, d['duration'])
                except: pass
            
            if all_lats:
                avg_ops = all_ops / max_dur
                p50 = np.percentile(all_lats, 50)
                p95 = np.percentile(all_lats, 95)
                p99 = np.percentile(all_lats, 99)
                
                print(f"  - Total Ops: {all_ops}")
                print(f"  - Avg Ops/sec: {avg_ops:.2f}")
                print(f"  - Latency (ms): P50={p50:.2f}, P95={p95:.2f}, P99={p99:.2f}")
                
                final_results[wl_id] = {
                    "ops_per_sec": avg_ops,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99
                }

    if rank == 0:
        output_path = REPO_ROOT / "benchmark" / "results" / f"mixed_workload_{system_name}_{job_id}.json"
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n✅ Benchmark Complete. Results saved to {output_path}")

    adapter.close()

if __name__ == "__main__":
    run_mixed_benchmark()
