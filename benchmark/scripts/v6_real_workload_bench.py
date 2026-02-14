#!/usr/bin/env python3
"""
Cascade V6 Real-World LLM Workload Benchmark
--------------------------------------------
Scenario:
1. Multi-tenant Shared Prefix: simulates same system prompt across all ranks.
2. Heavy Suffix Load: simulates unique user queries exceeding local GPU memory.
3. Hierarchical Overflow: observes cascading from GPU -> DRAM -> Remote -> Lustre.
4. Locality-Aware Retrieval: simulates repeat access to "overflowed" cache blocks.
"""

import sys
import os
import time
import hashlib
import numpy as np

# Path setup for Cascade library
default_build = '../../cascade_Code/cpp/build_mpi'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), build_dir)
sys.path.insert(0, build_dir)

try:
    import cascade_cpp
except ImportError:
    print("ERROR: cascade_cpp not found. Build with USE_MPI=ON first.")
    sys.exit(1)

def get_rank_info():
    rank = int(os.environ.get('SLURM_PROCID', os.environ.get('PMI_RANK', 0)))
    world = int(os.environ.get('SLURM_NTASKS', os.environ.get('PMI_SIZE', 1)))
    node = os.environ.get('SLURMD_NODENAME', 'localhost')
    return rank, world, node

def generate_block(layer, seq_start, size=160*1024):
    """Generate dummy KV data."""
    np.random.seed(layer * 1000 + seq_start)
    return np.random.randint(0, 255, size, dtype=np.uint8)

def print_rank0(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)

def main():
    rank, world, node = get_rank_info()
    
    # ─── Configuration ───
    # We set tight limits to force Tiering on Perlmutter
    cfg = cascade_cpp.DistributedConfig()
    cfg.num_gpus_per_node = 4
    cfg.gpu_capacity_per_device = 512 * 1024**2   # 512MB per GPU -> 2GB per node
    cfg.dram_capacity = 1024 * 1024**2            # 1GB DRAM per node
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.kv_compression = True
    cfg.promotion_threshold = 2
    
    if rank == 0:
        print("="*80)
        print(f" CASCADE V6 REAL WORKLOAD BENCHMARK (World: {world})")
        print("="*80)
        print(f" GPU Cap: {cfg.gpu_capacity_per_device/1024**2:.0f}MB/device")
        print(f" DRAM Cap: {cfg.dram_capacity/1024**2:.0f}MB/node")
        print(f" Novelties: Dedup={cfg.dedup_enabled}, Semantic={cfg.semantic_eviction}, Locality={cfg.locality_aware}")
        print("="*80)

    store = cascade_cpp.DistributedStore(cfg)
    block_size = 160 * 1024 # ~160KB

    # ─── Phase 1: Shared Prefix (Dedup & Semantic Protection) ───
    # Simulates 500 blocks of shared context (e.g. system prompt, long document)
    num_shared = 500 
    shared_keys = [f"prefix_block_{i:04d}" for i in range(num_shared)]
    shared_data = generate_block(0, 0, block_size) # Reuse same data for dedup efficiency

    t0 = time.time()
    for key in shared_keys:
        # All ranks write the same keys
        store.put(key, shared_data, is_prefix=True)
    store.barrier()
    t_phase1 = time.time() - t0
    
    # Sync metadata so everyone knows where prefix blocks are
    store.sync_metadata()
    store.barrier()

    stats = store.get_stats()
    print_rank0(f"[Phase 1] Shared Prefix Put: {t_phase1:.3f}s", rank)
    print_rank0(f"          Dedup Hits: {stats.dedup_hits} (saved {stats.dedup_bytes_saved/1024**2:.1f} MB)", rank)
    print_rank0(f"          Total Blocks: {stats.total_blocks}", rank)

    # ─── Phase 2: Heavy Load Overflow (Tiering Stress) ───
    # Each rank writes unique suffix blocks until local memory is full.
    # Node Capacity (GPU+DRAM) = 3GB approx. 
    # We write 20,000 blocks per node (approx 3.2GB)
    num_suffix = 20000 
    suffix_keys = [f"suffix_r{rank}_b{i:06d}" for i in range(num_suffix)]
    
    t0 = time.time()
    for key in suffix_keys:
        data = generate_block(rank, 1, block_size)
        store.put(key, data, is_prefix=False)
    store.barrier()
    t_phase2 = time.time() - t0

    # Sync metadata so Phase 3 knows where unique suffix blocks are
    store.sync_metadata()
    store.barrier()

    stats = store.get_stats()
    print_rank0(f"[Phase 2] Heavy Workload Put: {t_phase2:.3f}s", rank)
    print_rank0(f"          DRAM Evictions: {stats.dram_evictions}", rank)
    print_rank0(f"          Lustre Hits: {stats.lustre_hits}", rank)
    print_rank0(f"          Cluster GPU Used: {stats.cluster_gpu_used/1024**2:.1f} MB", rank)
    print_rank0(f"          Cluster DRAM Used: {stats.cluster_dram_used/1024**2:.1f} MB", rank)

    # ─── Phase 3: Locality-Aware Retrieval (Promotion) ───
    # Access remote blocks repeatedly to trigger promotion to local Tier 1/2.
    # We'll pick 100 blocks from "next" rank.
    target_rank = (rank + 1) % world
    remote_keys = [f"suffix_r{target_rank}_b{i:06d}" for i in range(100)]
    
    t0 = time.time()
    # Access 5 times (threshold is 2)
    for _ in range(5):
        for key in remote_keys:
            buf = np.empty(block_size, dtype=np.uint8)
            store.get(key, buf)
    store.barrier()
    t_phase3 = time.time() - t0

    stats = store.get_stats()
    print_rank0(f"[Phase 3] Locality-Aware Get: {t_phase3:.3f}s", rank)
    print_rank0(f"          Remote GPU Hits: {stats.remote_gpu_hits}", rank)
    print_rank0(f"          Remote DRAM Hits: {stats.remote_dram_hits}", rank)
    print_rank0(f"          Promotions to Local: {stats.promotions_to_local}", rank)

    # ─── Summary ───
    if rank == 0:
        print("\n" + "="*80)
        print(" FINAL RESULTS (Summary)")
        print("="*80)
        print(f" Throughput: {(num_suffix * block_size * world / t_phase2 / 1024**3):.2f} GB/s (Write)")
        print(f" Efficiency: Dedup saved {stats.dedup_bytes_saved/1024**2:.1f} MB CLUSTER-WIDE")
        print(f" Protection: {stats.prefix_blocks_protected} Prefix blocks avoided eviction")
        print(f" Promotions: {stats.promotions_to_local} Blocks moved to hot tiers via Locality-Awareness")
        print("="*80)

if __name__ == "__main__":
    main()
