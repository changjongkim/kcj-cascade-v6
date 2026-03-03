#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import argparse
import socket
from pathlib import Path

# MPI rank/world from SLURM
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

# Add repo root to path for adapters
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

# Import adapter factory
from benchmark.run_benchmark import get_adapter

def print_rank0(*args):
    if rank == 0: 
        print(*args, flush=True)

def clear_page_cache(file_path):
    """Evict file from OS page cache using posix_fadvise."""
    try:
        if os.path.isdir(file_path):
            for f in Path(file_path).glob("**/*"):
                if f.is_file(): clear_page_cache(str(f))
            return
        fd = os.open(file_path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except: pass

def run_recovery_profiling():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--block-size-mb", type=int, default=160)
    parser.add_argument("--num-blocks", type=int, default=20)
    args = parser.parse_args()

    # Total requests per node
    num_blocks = args.num_blocks
    block_size_bytes = args.block_size_mb * 1024 * 1024
    
    config = {}
    if args.system.lower() == "lmcache-redis":
        # Multi-node Redis host discovery
        rid = job_id
        tmp_h_dir = REPO_ROOT / "benchmark" / "tmp" / f"hosts_{rid}"
        redis_h_file = tmp_h_dir / "redis_host"
        
        # Wait for redis host file (up to 30s)
        for _ in range(30):
            if redis_h_file.exists():
                with open(redis_h_file, 'r') as f:
                    config["host"] = f.read().strip()
                    config["port"] = int(os.environ.get("REDIS_PORT", 6379))
                break
            time.sleep(1)
        
        if "host" not in config:
            # Fallback for single node or local test
            config["host"] = "localhost"
            config["port"] = int(os.environ.get("REDIS_PORT", 16379))
            
    adapter = get_adapter(args.system, config=config)
    if not adapter.initialize():
        print(f"Rank {rank}: Failed to initialize {args.system} on {config.get('host','localhost')}")
        return

    # 1. Prepare Data
    block_ids = [f"recovery_b{rank}_{i}" for i in range(num_blocks)]
    data = [np.random.randint(0, 256, block_size_bytes, dtype=np.uint8).tobytes() for _ in range(num_blocks)]
    
    print_rank0(f"\n🚀 Recovery Latency Profiling: {args.system} | {args.block_size_mb}MB blocks")
    print_rank0("="*70)
    print_rank0(f"{'Tier':<15} | {'Latency (ms)':>15} | {'Throughput (GB/s)':>20}")
    print_rank0("-" * 70)

    # ── Tier 1: Hot Recovery (Write & Immediate Read) ──
    # For Cascade, this is GPU -> GPU or GPU -> DRAM if no GPU
    # For others, this is Write -> OS Page Cache Read
    
    # Write Phase
    for i in range(num_blocks):
        mid = len(data[i]) // 2
        adapter.put(block_ids[i], data[i][:mid], data[i][mid:])
    
    adapter.flush()
    time.sleep(2)
    
    # Read Phase (Hot)
    t_start = time.time()
    for i in range(num_blocks):
        adapter.get(block_ids[i])
    t_hot = (time.time() - t_start) / num_blocks
    hot_bw = (block_size_bytes / 1024**3) / (t_hot if t_hot > 0 else 1e-9)
    print_rank0(f"{'HOT (HBM/Cache)':<15} | {t_hot*1000:15.2f} | {hot_bw:20.2f}")

    # ── Tier 2: Warm Recovery (Clear GPU/Volatile Cache) ──
    # For Cascade: clear_gpu_cache() -> Read from DRAM
    # For others: This behaves similarly to Hot unless they have a distinct DRAM layer
    if args.system.lower() == "cascade":
        try:
            # Access underlying store if possible to clear specific tier
            import cascade_cpp
            adapter.store.clear_tier(0) # Clear GPU
            time.sleep(2)
        except: pass
    
    t_start = time.time()
    for i in range(num_blocks):
        adapter.get(block_ids[i])
    t_warm = (time.time() - t_start) / num_blocks
    warm_bw = (block_size_bytes / 1024**3) / (t_warm if t_warm > 0 else 1e-9)
    print_rank0(f"{'WARM (DRAM/RDMA)':<15} | {t_warm*1000:15.2f} | {warm_bw:20.2f}")

    # ── Tier 3: Cold Recovery (Clear Everything / Disk) ──
    # Clear all caches and force read from Disk/Lustre
    if args.system.lower() == "cascade":
        try:
            adapter.store.clear_tier(0) # Clear GPU
            adapter.store.clear_tier(1) # Clear DRAM
            lustre_path = adapter.lustre_path if hasattr(adapter, 'lustre_path') else f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/cascade_store"
            clear_page_cache(lustre_path)
        except: pass
    else:
        # Baselines: Just clear OS page cache
        # Each adapter sets its own storage directory. We try to read it dynamically.
        storage_path = None
        if hasattr(adapter, 'storage_path'):
            storage_path = adapter.storage_path
        elif hasattr(adapter, 'base_dir'):
            storage_path = str(adapter.base_dir)
        elif hasattr(adapter, 'file_path'):
            storage_path = adapter.file_path
        
        if storage_path and os.path.exists(storage_path):
            clear_page_cache(storage_path)
            
    # Extra protection: wait to ensure OS flushes if it was asynchronous
    time.sleep(5)

    t_start = time.time()
    for i in range(num_blocks):
        adapter.get(block_ids[i])
    t_cold = (time.time() - t_start) / num_blocks
    cold_bw = (block_size_bytes / 1024**3) / (t_cold if t_cold > 0 else 1e-9)
    print_rank0(f"{'COLD (Disk/Lustre)':<15} | {t_cold*1000:15.2f} | {cold_bw:20.2f}")

    print_rank0("="*70)
    print_rank0("✅ Success")
    
    adapter.close()

if __name__ == "__main__":
    run_recovery_profiling()
