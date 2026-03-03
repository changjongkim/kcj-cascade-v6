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

def run_index_scalability_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--test-reqs", type=int, default=128, help="Number of concurrent read requests to measure latency/TTFT")
    parser.add_argument("--block-size", type=int, default=16, help="Size in MB of each block")
    args = parser.parse_args()

    system_name = args.system
    
    # Config setup
    config = {}
    if system_name.lower() == "cascade":
        config = {"gpu_capacity_gb": 36.0, "shm_capacity_gb": 128.0, "use_gpu": True, "use_compression": False}
    elif "redis" in system_name.lower():
        r_port = int(os.environ.get("REDIS_PORT", 16379))
        shared_redis_host = "localhost"
        tmp_h_dir = REPO_ROOT / "benchmark" / "tmp" / f"hosts_{job_id}"
        
        if system_name.lower() == "redis-dist":
            # Read all hostnames for sharding
            all_hosts_path = tmp_h_dir / "all_hosts"
            if all_hosts_path.exists():
                with open(all_hosts_path, 'r') as f:
                    hosts = [line.strip() for line in f if line.strip()]
                config = {"hosts": hosts, "port": r_port}
            else:
                config = {"hosts": ["localhost"], "port": r_port}
        else:
            # Single node Redis
            if tmp_h_dir.exists() and (tmp_h_dir / "redis_host").exists():
                with open(tmp_h_dir / "redis_host", 'r') as f:
                    shared_redis_host = f.read().strip()
            config = {"host": shared_redis_host, "port": r_port}
    elif system_name.lower() == "lmcache":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/benchmark/lmcache_idx_{job_id}"}
    elif system_name.lower() == "pdc":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/benchmark/pdc_idx_{job_id}"}
    elif "hdf5" in system_name.lower():
        config = {"file_path": f"/pscratch/sd/s/sgkim/kcj/benchmark/tmp/h5_idx_{job_id}.h5", "use_mpi": True}
    elif system_name.lower() == "llm-gpu":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/benchmark/llmgpu_idx_{job_id}"}

    adapter = get_adapter(system_name, config)
    if not adapter.initialize():
        print(f"❌ [{system_name}] Failed to initialize on Rank {rank}")
        return

    # Scale steps: Total blocks in the system
    scale_steps = [1000, 10000, 50000] 
    
    block_size_bytes = args.block_size * 1024 * 1024
    
    # Pre-generate dummy data 
    dummy_data = b"x" * block_size_bytes
    dummy_key_prefix = b"k" * 128
    
    results = {}
    current_total = 0

    for target_total in scale_steps:
        print_rank0(f"\n▶ Scaling to {target_total} blocks ({args.block_size}MB each)")
        
        print_rank0(f"  - Starting Ingestion...")
        for i in range(current_total, target_total):
            if i % world == rank:
                key = f"idx_scale_blk_{i}"
                adapter.put(key, dummy_key_prefix, dummy_data)
        
        adapter.flush()
        current_total = target_total
        time.sleep(5) # Sync wait
        
        print_rank0(f"  - Measuring Get Latency/BW with {args.test_reqs} concurrent equivalent requests...")
        
        reqs_per_rank = args.test_reqs // world
        latencies = []
        test_indices = np.random.randint(0, current_total, reqs_per_rank)
        
        # Actual Get Phase
        phase_start = time.perf_counter()
        for idx in test_indices:
            key = f"idx_scale_blk_{idx}"
            t0 = time.perf_counter()
            res = adapter.get(key)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # ms
        phase_end = time.perf_counter()
            
        # Write temporary results for aggregation
        res_dir = REPO_ROOT / "benchmark" / "tmp" / f"scale_v3_{job_id}_{target_total}"
        res_dir.mkdir(parents=True, exist_ok=True)
        with open(res_dir / f"rank_{rank}.json", "w") as f:
            json.dump({
                "lats": latencies,
                "start": phase_start,
                "end": phase_end,
                "count": len(test_indices)
            }, f)
            
        time.sleep(5) # wait for all ranks to write files
        
        if rank == 0:
            all_lats = []
            total_mb_all = 0
            agg_bw_sum = 0
            
            for r in range(world):
                try:
                    with open(res_dir / f"rank_{r}.json", "r") as f:
                        data = json.load(f)
                        all_lats.extend(data['lats'])
                        
                        rank_time = data['end'] - data['start']
                        rank_mb = data['count'] * args.block_size
                        if rank_time > 0:
                            agg_bw_sum += (rank_mb / rank_time)
                        
                        total_mb_all += rank_mb
                except Exception:
                    pass
                    
            if all_lats:
                avg_lat = np.mean(all_lats)
                p50 = np.percentile(all_lats, 50)
                p95 = np.percentile(all_lats, 95)
                p99 = np.percentile(all_lats, 99)
                
                print_rank0(f"  - Results: P50={p50:.2f}ms, P99={p99:.2f}ms, TTFT Proxy={p95:.2f}ms")
                print_rank0(f"  - Aggregated Bandwidth (Sum of Rank BWs): {agg_bw_sum:.2f} MB/s (Total {total_mb_all} MB)")
                
                results[target_total] = {
                    "avg_ms": avg_lat,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "agg_bw_mbs": agg_bw_sum
                }

    if rank == 0:
        output_path = REPO_ROOT / "benchmark" / "results" / f"scale_realistic_agg_{system_name}_{job_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Complete. Results saved to {output_path}")

    adapter.close()

if __name__ == "__main__":
    run_index_scalability_benchmark()
