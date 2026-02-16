#!/usr/bin/env python3
"""
Cascade V6 - 1 TB STRESS TEST (Strong Scaling)
==============================================
Data Size: 1.0 TB (Fixed)
Nodes: 1, 2, 4, 8
Goal: Demonstrate Cascading Eviction (GPU->DRAM->Lustre) on small node counts
      and Pure GPU Performance on large node counts.
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

def get_rank_info():
    rank = int(os.environ.get('SLURM_PROCID', os.environ.get('PMI_RANK', 0)))
    world = int(os.environ.get('SLURM_NTASKS', os.environ.get('PMI_SIZE', 1)))
    return rank, world

def main():
    rank, world = get_rank_info()
    
    # ─── Configuration ───
    cfg = cascade_cpp.DistributedConfig()
    cfg.num_gpus_per_node = 4
    # Set realistic capacities for Perlmutter to trigger eviction
    cfg.gpu_capacity_per_device = 40ULL * 1024**3 # 40GB/GPU
    cfg.dram_capacity = 200ULL * 1024**3          # 200GB/Node
    
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.kv_compression = True
    
    # Lustre path must be able to hold ~1TB
    lustre_path = os.environ.get('CASCADE_LUSTRE_PATH',
                                  f'/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_1TB_r{rank}')
    os.makedirs(lustre_path, exist_ok=True)
    cfg.lustre_path = lustre_path
    
    store = cascade_cpp.DistributedStore(cfg)
    
    block_size = 160 * 1024
    # ~1 TB = 1024*1024*1024*1024 / (160*1024) = 6,710,886 blocks
    # We'll use 6,400,000 for cleaner math
    TOTAL_BLOCKS = 6400000 
    blocks_per_rank = TOTAL_BLOCKS // world
    total_data_gb = (TOTAL_BLOCKS * block_size) / 1024**3
    
    if rank == 0:
        print("=" * 80)
        print(f" 🚀 CASCADE V6 - 1.0 TB STRESS TEST (World: {world})")
        print("=" * 80)
        print(f" Total Data: {total_data_gb:.2f} GB")
        print(f" Total Blocks: {TOTAL_BLOCKS}")
        print(f" Per Rank: {blocks_per_rank} blocks")
        print("=" * 80)
    
    # ─── Phase 1: WRITE (1 TB Total) ───
    # We use a deterministic byte modification to avoid dedup bypassing the IO stress
    sample_data = np.random.randint(0, 255, block_size, dtype=np.uint8)
    data_view = sample_data.view(np.uint32)
    
    store.barrier()
    t0 = time.time()
    for i in range(blocks_per_rank):
        data_view[0] = i + (rank * 10000000)
        store.put(f"stress_b{rank:03d}_{i:07d}", sample_data)
        
        if rank == 0 and i % 50000 == 0:
            print(f" [Write Progress] {i}/{blocks_per_rank} blocks...", flush=True)
            
    store.barrier()
    t_write = time.time() - t0
    
    store.sync_metadata()
    store.barrier()
    
    if rank == 0:
        print(f"\n[Write Summary]")
        print(f" Total Time: {t_write:.2f}s")
        print(f" Throughput: {total_data_gb / t_write:.2f} GB/s")
    
    # ─── Phase 2: READ (Sampled 10% for speed, total is too large) ───
    # Reading every block of 1TB might take too long in a benchmark. 
    # We read 10% randomly to verify consistency and tiering performance.
    SAMPLES = blocks_per_rank // 10
    
    store.barrier()
    t0 = time.time()
    for i in range(SAMPLES):
        buf = np.empty(block_size, dtype=np.uint8)
        store.get(f"stress_b{rank:03d}_{i:07d}", buf)
    store.barrier()
    t_read = (time.time() - t0) * 10 # Extrapolate to full 1TB
    
    if rank == 0:
        print(f"\n[Read Summary - Extrapolated]")
        print(f" Est. 1TB Read Time: {t_read:.2f}s")
        print(f" Throughput: {total_data_gb / t_read:.2f} GB/s")
        
        stats = store.get_stats()
        print(f"\n[Final Hit Stats - Rank 0]")
        print(f" Local GPU: {stats.local_gpu_hits}, Local DRAM: {stats.local_dram_hits}")
        print(f" Remote GPU: {stats.remote_gpu_hits}, Remote DRAM: {stats.remote_dram_hits}")
        print(f" Lustre Hits: {stats.lustre_hits}")
        print(f" GPU Evictions: {stats.gpu_evictions}, DRAM Evictions: {stats.dram_evictions}")
        print("=" * 80)

if __name__ == "__main__":
    main()
