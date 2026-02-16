#!/usr/bin/env python3
"""
Cascade V6 - 5-Tier Full Hierarchy Benchmark
==========================================
Exercises all 5 Tiers: Local GPU, Local DRAM, Remote GPU, Remote DRAM, Lustre.
Modes: 
  - Weak Scaling: Work per node fixed (e.g., 500GB/node)
  - Strong Scaling: Total work fixed (e.g., 1TB total)
Pattern: Random Read to force Cross-Node RDMA.
"""

import sys
import os
import time
import numpy as np

# Path setup
default_build = '../../cascade_Code/cpp/build_cascade_cpp'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), build_dir)
sys.path.insert(0, build_dir)

try:
    import cascade_cpp
except ImportError:
    print("ERROR: cascade_cpp not found.")
    sys.exit(1)

def get_unique_block(rank, seq_id, size=160*1024):
    """Generate unique data for every block to prevent unintended dedup hits."""
    # Use a faster way than full random for huge datasets
    data = np.full(size, rank % 255, dtype=np.uint8)
    # Inject rank and sequence ID into the first 8 bytes
    view = data.view(np.uint64)
    view[0] = (np.uint64(rank) << np.uint64(32)) | np.uint64(seq_id)
    return data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["weak", "strong"], required=True)
    parser.add_argument("--total-gb", type=float, default=1024.0, help="Total data in GB for Strong Scaling")
    parser.add_argument("--node-gb", type=float, default=500.0, help="Data per node in GB for Weak Scaling")
    args = parser.parse_args()

    rank = int(os.environ.get('SLURM_PROCID', 0))
    world = int(os.environ.get('SLURM_NTASKS', 1))
    
    # ─── Configuration ───
    cfg = cascade_cpp.DistributedConfig()
    cfg.num_gpus_per_node = 4  # 4 GPUs/node, each rank owns 1 (local_rank % 4)
    cfg.gpu_capacity_per_device = 38 * 1024**3  # 38GB per GPU (A100 40GB with headroom)
    cfg.dram_capacity = 200 * 1024**3           # 200GB DRAM (1 rank per node, ~256GB total)
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.kv_compression = True # Compression on for realistic tiering
    
    lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_5tier_r{rank}"
    os.makedirs(lustre_path, exist_ok=True)
    cfg.lustre_path = lustre_path
    
    store = cascade_cpp.DistributedStore(cfg)
    block_size = 160 * 1024 # 160KB
    
    # Calculation
    if args.mode == "strong":
        TOTAL_BLOCKS = int((args.total_gb * 1024**3) / block_size)
        my_blocks = TOTAL_BLOCKS // world
    else:
        my_blocks = int((args.node_gb * 1024**3) / block_size)
        TOTAL_BLOCKS = my_blocks * world
        
    total_data_gb = (TOTAL_BLOCKS * block_size) / 1024**3
    
    if rank == 0:
        print("=" * 80)
        print(f" 🏔️  CASCADE V6 - 5-TIER {args.mode.upper()} SCALING")
        print("=" * 80)
        print(f" Mode:           {args.mode}")
        print(f" Total Data:     {total_data_gb:.2f} GB")
        print(f" Blocks/Rank:    {my_blocks}")
        print(f" Nodes:          {world}")
        print("=" * 80)

    # ─── Phase 1: Write (Tier 1, 2, 5 Exercise) ───
    # We use uint64 packing for speed
    import struct
    
    sample_data = np.zeros(block_size, dtype=np.uint8)
    data_view = sample_data.view(np.uint64)
    
    store.barrier()
    t0 = time.time()
    for i in range(my_blocks):
        # Unique data content
        data_view[0] = (int(rank) << 32) | i
        # Successive calls to store.put
        store.put(f"b{rank:03d}_{i:08d}", sample_data)
        if rank == 0 and i % 50000 == 0:
            print(f" [Write] Progress: {i}/{my_blocks}...", flush=True)
            
    store.barrier()
    t_write = time.time() - t0
    
    store.sync_metadata()
    store.barrier()
    
    # ─── Phase 2: RANDOM Read (Tier 3, 4 Exercise) ───
    # Mix of local and remote reads
    SAMPLES = min(my_blocks, 10000) # Read enough, but not forever
    np.random.seed(rank + 42)
    
    # Pick random keys from ANY rank
    read_indices = np.random.randint(0, world, SAMPLES)
    read_seqs = np.random.randint(0, my_blocks, SAMPLES)
    
    store.barrier()
    t0 = time.time()
    for i in range(SAMPLES):
        r = read_indices[i]
        s = read_seqs[i]
        buf = np.empty(block_size, dtype=np.uint8)
        store.get(f"b{r:03d}_{s:08d}", buf)
        
    store.barrier()
    t_read = time.time() - t0
    
    # Stats aggregation
    stats = store.get_stats()
    
    if rank == 0:
        print(f"\n[Summary - {args.mode.upper()}]")
        print(f" Nodes:          {world // 4}")
        print(f" Write Time:     {t_write:.2f}s ({total_data_gb / t_write:.2f} GB/s)")
        print(f" Read (Sample):  {t_read:.5f}s (Avg { (t_read/SAMPLES)*1000:.2f} ms/block)")
        print(f"\n[Tiering Efficiency - Rank 0]")
        print(f" ├─ Local GPU:   {stats.local_gpu_hits}")
        print(f" ├─ Local DRAM:  {stats.local_dram_hits}")
        print(f" ├─ Remote GPU:  {stats.remote_gpu_hits}")
        print(f" ├─ Remote DRAM: {stats.remote_dram_hits}")
        print(f" └─ Lustre:      {stats.lustre_hits}")
        print(f" Evictions:      GPU={stats.gpu_evictions}, DRAM={stats.dram_evictions}")
        print("=" * 80)

if __name__ == "__main__":
    main()
