#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import argparse
from pathlib import Path
import subprocess

# Add repo root to path for adapters
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

from benchmark.run_benchmark import get_adapter

# MPI rank/world from SLURM
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

def print_rank0(*args):
    if rank == 0: 
        print(*args, flush=True)

def clear_page_cache(path):
    """Evict directory/file from OS page cache using a bash command"""
    try:
        # Run vmtouch -e on the directory to clear OS page cache
        subprocess.run(["vmtouch", "-e", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        pass

def file_barrier(name):
    bar_dir = REPO_ROOT / "benchmark" / "tmp" / f"bar_c2h_{job_id}_{name}"
    bar_dir.mkdir(parents=True, exist_ok=True)
    (bar_dir / f"rank_{rank}").touch()
    while True:
        try:
            if len(list(bar_dir.iterdir())) >= world: break
        except: pass
        time.sleep(0.5)

def run_cold_to_hot_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="cascade")
    parser.add_argument("--num-blocks", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=16, help="Size in MB of each block")
    args = parser.parse_args()

    block_size_bytes = args.block_size * 1024 * 1024
    
    config = {
        "gpu_capacity_gb": 36.0, 
        "shm_capacity_gb": 128.0, 
        "use_gpu": True, 
        "use_compression": False,
        "lustre_path": f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/c2h_{job_id}"
    }

    adapter = get_adapter(args.system, config=config)
    if not adapter.initialize():
        print_rank0("Failed to initialize adapter")
        return

    print_rank0(f"\n=======================================================")
    print_rank0(f"❄️🔥 Cascade Cold-to-Hot Promotion Benchmark")
    print_rank0(f"   Nodes: {world} | Blocks: {args.num_blocks} ({args.block_size}MB each)")
    print_rank0(f"=======================================================\n")

    # 1. Write Phase (Inject to system)
    print_rank0("[Phase 1] Writing initial data into system...")
    base_data = np.random.randint(0, 256, block_size_bytes, dtype=np.uint8)
    dummy_key_prefix = b"k" * 128
    
    my_keys = []
    # Distribute writes across ranks
    for i in range(args.num_blocks):
        if i % world == rank:
            key = f"c2h_blk_{i}"
            my_keys.append(key)
            current_block = base_data.copy()
            current_block[:8] = np.frombuffer(i.to_bytes(8, 'little'), dtype=np.uint8)
            adapter.put(key, dummy_key_prefix, current_block.tobytes())
            
    adapter.flush()
    if hasattr(adapter, "sync_metadata"): adapter.sync_metadata()
    file_barrier("write_done")
    
    # 2. Make it COLD (Drop Caches)
    print_rank0("[Phase 2] Forcing Eviction to simulate COLD start (Dropping RAM/GPU caches)...")
    if hasattr(adapter, 'store') and hasattr(adapter.store, 'clear_tier'):
        try:
            adapter.store.clear_tier(0) # Clear GPU
            adapter.store.clear_tier(1) # Clear DRAM
        except Exception as e:
            print_rank0(f"Failed to clear Cascade tier: {e}")
            
    clear_page_cache(config["lustre_path"])
    file_barrier("cache_dropped")
    
    # Generate read workload (Rank 0 reads its own blocks for simplicity to measure straight BW)
    # We will measure total aggregated bandwidth

    def measure_read_phase(phase_name):
        print_rank0(f"[{phase_name}] Performing 128-thread concurrent reads...")
        test_indices = np.random.choice(my_keys, size=min(128, len(my_keys)), replace=True) if len(my_keys) > 0 else []
        
        file_barrier(f"start_{phase_name}")
        
        hits = 0
        t0 = time.perf_counter()
        for k in test_indices:
            # We want to measure the exact time including tensor copy overhead
            t_s = time.perf_counter()
            res = adapter.get(k)
            if res:
                hits += 1
                _ = bytes(res[0])
                _ = bytes(res[1])
        t1 = time.perf_counter()
        
        duration = t1 - t0
        mb_read = hits * args.block_size
        
        res_dir = REPO_ROOT / "benchmark" / "tmp" / f"c2h_res_{job_id}_{phase_name}"
        res_dir.mkdir(parents=True, exist_ok=True)
        with open(res_dir / f"rank_{rank}.json", "w") as f:
            json.dump({"mb_read": mb_read, "duration": duration, "hits": hits}, f)
            
        file_barrier(f"end_{phase_name}")
        
        if rank == 0:
            total_mb = 0
            total_bw = 0
            total_hits = 0
            for r in range(world):
                try:
                    with open(res_dir / f"rank_{r}.json", "r") as f:
                        data = json.load(f)
                        r_mb = data["mb_read"]
                        r_dur = data["duration"]
                        total_mb += r_mb
                        total_hits += data["hits"]
                        if r_dur > 0:
                            total_bw += (r_mb / r_dur)
                except Exception: pass
            
            gb_s = total_bw / 1024
            print_rank0(f"   => {phase_name} Results: {total_hits} hits, Aggregated Bandwidth: {gb_s:.2f} GB/s")
            return gb_s
            
    # 3. Cold Start Read
    cold_bw = measure_read_phase("Phase 3 (COLD Read)")
    
    # 4. Hot Start Read (Data should now be promoted)
    hot_bw = measure_read_phase("Phase 4 (HOT Read)")

    if rank == 0:
        print_rank0("\n📊 SUMMARY: Cold Promotion to Hot")
        print_rank0("-" * 40)
        print_rank0(f"  🧊 Cold Start (Lustre):   {cold_bw:8.2f} GB/s")
        print_rank0(f"  🔥 Hot Promoted (Memory): {hot_bw:8.2f} GB/s")
        print_rank0(f"  🚀 Speedup Factor:        {hot_bw/cold_bw:.1f}x")
        print_rank0("-" * 40)
        
    adapter.close()

if __name__ == "__main__":
    run_cold_to_hot_benchmark()
