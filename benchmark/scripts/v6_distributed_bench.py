#!/usr/bin/env python3
"""
Cascade V6 Distributed Benchmark — Multi-Node Multi-GPU

Tests the 3 core novelties:
  1. Cross-node semantic-aware eviction (prefix block protection)
  2. Distributed content-addressed deduplication (SHA256 global index)
  3. Locality-aware hierarchical placement (access frequency tracking)

Run: srun -N2 -n8 --gpus-per-node=4 python v6_distributed_bench.py
"""

import sys
import os
import time
import hashlib
import numpy as np

# Add MPI build directory first (must load before non-MPI build)
default_build = '../../cascade_Code/cpp/build_mpi'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), build_dir)
sys.path.insert(0, build_dir)

try:
    import cascade_cpp
    HAS_DISTRIBUTED = hasattr(cascade_cpp, 'DistributedStore')
except ImportError:
    print("ERROR: cascade_cpp not found. Build with USE_MPI=ON first.")
    sys.exit(1)


def get_rank_info():
    """Get MPI rank info from environment or cascade_cpp."""
    rank = int(os.environ.get('SLURM_PROCID', os.environ.get('PMI_RANK', 0)))
    world = int(os.environ.get('SLURM_NTASKS', os.environ.get('PMI_SIZE', 1)))
    node = os.environ.get('SLURMD_NODENAME', 'localhost')
    return rank, world, node


def generate_kv_block(layer, head, seq_start, seq_len=128, hidden=128, dtype=np.float16):
    """Generate a realistic KV cache block (FP16)."""
    np.random.seed(layer * 1000 + head * 100 + seq_start)
    k = np.random.randn(seq_len, hidden).astype(dtype)
    v = np.random.randn(seq_len, hidden).astype(dtype)
    return np.concatenate([k.ravel(), v.ravel()]).view(np.uint8)


def block_id(layer, head, seq_start, prefix=""):
    """Generate SHA256-based block ID."""
    key = f"{prefix}:L{layer}:H{head}:S{seq_start}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def print_rank0(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)


# ============================================================================
# Test 1: Distributed Dedup (Novelty 2)
# ============================================================================
def test_distributed_dedup(rank, world):
    """
    Multiple ranks write the same prefix blocks → only stored once.
    Each rank writes its own unique suffix blocks.
    """
    print_rank0(f"\n{'='*70}", rank)
    print_rank0(f"  Test 1: Distributed Content-Addressed Deduplication", rank)
    print_rank0(f"{'='*70}", rank)

    cfg = cascade_cpp.DistributedConfig()
    cfg.gpu_capacity_per_device = 2 * 1024**3   # 2 GB per GPU
    cfg.dram_capacity = 4 * 1024**3              # 4 GB per node
    cfg.num_gpus_per_node = 4
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = False  # Disable for this test

    store = cascade_cpp.DistributedStore(cfg)

    # Shared prefix: same system prompt KV cache (all ranks write same blocks)
    num_prefix_blocks = 20
    prefix_blocks = []
    for i in range(num_prefix_blocks):
        bid = block_id(0, 0, i, prefix="shared_system_prompt")
        data = generate_kv_block(0, 0, i)
        prefix_blocks.append((bid, data))

    # Each rank writes all prefix blocks
    t0 = time.time()
    for bid, data in prefix_blocks:
        store.put(bid, data, is_prefix=True)
    store.barrier()
    
    # CRITICAL: Sync metadata after writing prefix blocks so other nodes know about them
    store.sync_metadata()
    store.barrier()
    
    # NOW: Write the same prefix blocks again → should trigger dedup hits
    for bid, data in prefix_blocks:
        store.put(bid, data, is_prefix=True)
    store.barrier()
    
    t_prefix = time.time() - t0

    stats = store.get_stats()
    print_rank0(f"\n  [Prefix Write + Dedup] {num_prefix_blocks} blocks × {world} ranks × 2 rounds", rank)
    print_rank0(f"    Time: {t_prefix:.3f}s", rank)
    print_rank0(f"    Dedup hits: {stats.dedup_hits}", rank)
    print_rank0(f"    Dedup bytes saved: {stats.dedup_bytes_saved / 1024**2:.1f} MB", rank)
    print_rank0(f"    Prefix blocks: {stats.prefix_blocks}", rank)

    # Unique suffix: each rank writes its own KV cache
    num_suffix_blocks = 30
    t0 = time.time()
    for i in range(num_suffix_blocks):
        bid = block_id(0, 0, 100 + i, prefix=f"rank{rank}_query")
        data = generate_kv_block(0, 0, 100 + i)
        store.put(bid, data, is_prefix=False)
    store.barrier()
    t_suffix = time.time() - t0

    stats = store.get_stats()
    print_rank0(f"\n  [Suffix Write] {num_suffix_blocks} blocks × {world} ranks (unique)", rank)
    print_rank0(f"    Time: {t_suffix:.3f}s", rank)
    print_rank0(f"    Total blocks: {stats.total_blocks}", rank)
    print_rank0(f"    Dedup hits total: {stats.dedup_hits}", rank)

    # Read back — all ranks should be able to read prefix blocks
    t0 = time.time()
    hits = 0
    for bid, data in prefix_blocks:
        buf = np.empty(data.shape, dtype=np.uint8)
        found, size = store.get(bid, buf)
        if found:
            hits += 1
    store.barrier()
    t_read = time.time() - t0

    print_rank0(f"\n  [Prefix Read] {hits}/{num_prefix_blocks} blocks found", rank)
    print_rank0(f"    Time: {t_read:.3f}s", rank)

    return stats


# ============================================================================
# Test 2: Semantic Eviction (Novelty 1)
# ============================================================================
def test_semantic_eviction(rank, world):
    """
    Fill DRAM with blocks, then add more to trigger eviction.
    Prefix blocks should be protected; non-prefix blocks evicted first.
    """
    print_rank0(f"\n{'='*70}", rank)
    print_rank0(f"  Test 2: Cross-Node Semantic-Aware Eviction", rank)
    print_rank0(f"{'='*70}", rank)

    block_size = 2 * 1024 * 1024  # 2MB blocks

    cfg = cascade_cpp.DistributedConfig()
    cfg.gpu_capacity_per_device = 5 * 1024**2   # 5 MB per GPU
    cfg.dram_capacity = 20 * 1024**2             # 20 MB DRAM
    cfg.num_gpus_per_node = 1
    cfg.dedup_enabled = False
    cfg.semantic_eviction = True
    cfg.locality_aware = False

    store = cascade_cpp.DistributedStore(cfg)

    # Write 10 prefix blocks (should be protected)
    num_prefix = 10
    prefix_ids = []
    for i in range(num_prefix):
        bid = f"prefix_{i:04d}"
        data = np.random.randint(0, 255, block_size, dtype=np.uint8)
        store.put(bid, data, is_prefix=True)
        prefix_ids.append(bid)

    # Write 40 non-prefix blocks → will force eviction (total 100MB > 25MB)
    num_regular = 40
    for i in range(num_regular):
        bid = f"regular_rank{rank}_{i:04d}"
        data = np.random.randint(0, 255, block_size, dtype=np.uint8)
        store.put(bid, data, is_prefix=False)

    store.barrier()
    stats = store.get_stats()

    print_rank0(f"\n  Wrote {num_prefix} prefix + {num_regular} regular blocks (1MB each)", rank)
    print_rank0(f"  DRAM capacity: 50MB, GPU: 10MB → eviction required", rank)
    print_rank0(f"  DRAM evictions: {stats.dram_evictions}", rank)
    print_rank0(f"  Prefix blocks protected: {stats.prefix_blocks_protected}", rank)
    print_rank0(f"  Prefix blocks in registry: {stats.prefix_blocks}", rank)

    # Check: can we still read prefix blocks?
    prefix_hits = 0
    for bid in prefix_ids:
        buf = np.empty(block_size, dtype=np.uint8)
        found, size = store.get(bid, buf)
        if found:
            prefix_hits += 1

    print_rank0(f"  Prefix block readback: {prefix_hits}/{num_prefix} found ✓" if prefix_hits == num_prefix else f"  Prefix block readback: {prefix_hits}/{num_prefix} ✗", rank)

    return stats


# ============================================================================
# Test 3: Locality-Aware Placement (Novelty 3)
# ============================================================================
def test_locality_aware(rank, world):
    """
    Rank 0 repeatedly accesses a block stored on another node.
    After promotion_threshold accesses, block should be promoted to local GPU.
    """
    print_rank0(f"\n{'='*70}", rank)
    print_rank0(f"  Test 3: Locality-Aware Hierarchical Placement", rank)
    print_rank0(f"{'='*70}", rank)

    cfg = cascade_cpp.DistributedConfig()
    cfg.gpu_capacity_per_device = 2 * 1024**3
    cfg.dram_capacity = 4 * 1024**3
    cfg.num_gpus_per_node = 4
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.promotion_threshold = 3

    store = cascade_cpp.DistributedStore(cfg)

    # Each rank writes 20 blocks
    my_blocks = []
    for i in range(20):
        bid = block_id(rank, 0, i)
        data = generate_kv_block(rank, 0, i)
        store.put(bid, data)
        my_blocks.append((bid, data))

    store.barrier()
    store.sync_metadata() # Novelty 3 needs to know where remote blocks are
    store.barrier()

    # Rank 0 repeatedly reads blocks from rank 1's set
    num_accesses = 10
    t0 = time.time()
    for access_round in range(num_accesses):
        # Each rank reads some of its own blocks (local) + some from "next" rank
        for i in range(5):
            bid = block_id((rank + 1) % world, 0, i)
            buf = np.empty(generate_kv_block(0, 0, 0).shape, dtype=np.uint8)
            store.get(bid, buf)

    store.barrier()
    t_total = time.time() - t0

    stats = store.get_stats()
    print_rank0(f"\n  {num_accesses} rounds × 5 cross-rank reads", rank)
    print_rank0(f"  Total time: {t_total:.3f}s", rank)
    print_rank0(f"  Local GPU hits: {stats.local_gpu_hits}", rank)
    print_rank0(f"  Local DRAM hits: {stats.local_dram_hits}", rank)
    print_rank0(f"  Remote GPU hits: {stats.remote_gpu_hits}", rank)
    print_rank0(f"  Remote DRAM hits: {stats.remote_dram_hits}", rank)
    print_rank0(f"  Promotions to local: {stats.promotions_to_local}", rank)
    print_rank0(f"  Promotion threshold: {cfg.promotion_threshold}", rank)

    return stats


# ============================================================================
# Test 4: Scaling throughput (strong scaling)
# ============================================================================
def test_scaling(rank, world):
    """
    Measure aggregate write/read throughput across all ranks.
    """
    print_rank0(f"\n{'='*70}", rank)
    print_rank0(f"  Test 4: Multi-Node Scaling ({world} ranks)", rank)
    print_rank0(f"{'='*70}", rank)

    cfg = cascade_cpp.DistributedConfig()
    cfg.gpu_capacity_per_device = 4 * 1024**3
    cfg.dram_capacity = 8 * 1024**3
    cfg.num_gpus_per_node = 4
    cfg.dedup_enabled = False
    cfg.semantic_eviction = False
    cfg.locality_aware = False
    cfg.kv_compression = True

    store = cascade_cpp.DistributedStore(cfg)

    # Pre-generate data to avoid measuring Python random/numpy overhead
    num_blocks = 500
    block_size = 160 * 1024
    test_data = [np.random.randint(0, 255, block_size, dtype=np.uint8) for _ in range(num_blocks)]
    test_keys = [f"scale_r{rank}_b{i:06d}" for i in range(num_blocks)]
    read_bufs = [np.empty(block_size, dtype=np.uint8) for _ in range(num_blocks)]

    # Write
    t0 = time.time()
    for i in range(num_blocks):
        store.put(test_keys[i], test_data[i])
    store.barrier()
    t_write = time.time() - t0

    total_bytes = num_blocks * block_size * world
    write_gbps = total_bytes / t_write / 1024**3

    # Read
    t0 = time.time()
    for i in range(num_blocks):
        store.get(test_keys[i], read_bufs[i])
    store.barrier()
    t_read = time.time() - t0

    read_gbps = total_bytes / t_read / 1024**3

    stats = store.get_stats()
    print_rank0(f"\n  {num_blocks} blocks × {world} ranks × {block_size//1024}KB = {total_bytes/1024**2:.0f} MB total", rank)
    print_rank0(f"  Write: {write_gbps:.2f} GB/s aggregate ({t_write:.3f}s)", rank)
    print_rank0(f"  Read:  {read_gbps:.2f} GB/s aggregate ({t_read:.3f}s)", rank)
    print_rank0(f"  Compression savings: {stats.compression_savings / 1024**2:.1f} MB", rank)
    print_rank0(f"  Cluster GPU used: {stats.cluster_gpu_used / 1024**2:.1f} MB", rank)
    print_rank0(f"  Cluster DRAM used: {stats.cluster_dram_used / 1024**2:.1f} MB", rank)

    return {'write_gbps': write_gbps, 'read_gbps': read_gbps}


# ============================================================================
# Main
# ============================================================================
def main():
    rank, world, node = get_rank_info()
    
    if rank == 0:
        print("╔════════════════════════════════════════════════════════════╗")
        print("║   Cascade V6 Distributed Benchmark — 3 Novelties          ║")
        print("║   Cross-Node Semantic Eviction | Distributed Dedup        ║")
        print("║   Locality-Aware Placement | 5-Tier Hierarchy             ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"  Ranks: {world} | Node: {node}")
        print(f"  Has DistributedStore: {HAS_DISTRIBUTED}")

    if not HAS_DISTRIBUTED:
        if rank == 0:
            print("\n⚠️  DistributedStore not available (build with USE_MPI=ON)")
            print("  Running single-node simulation instead...")

        # Single-node fallback: test non-MPI path
        test_single_node_simulation(rank, world)
        return

    # Run all 4 tests
    dedup_stats = test_distributed_dedup(rank, world)
    evict_stats = test_semantic_eviction(rank, world)
    locality_stats = test_locality_aware(rank, world)
    scaling = test_scaling(rank, world)

    # Summary
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Summary: Cascade V6 Distributed ({world} ranks)")
        print(f"{'='*70}")
        print(f"  Novelty 1 — Semantic Eviction:")
        print(f"    DRAM evictions: {evict_stats.dram_evictions}")
        print(f"    Prefix protected: {evict_stats.prefix_blocks}")
        print(f"  Novelty 2 — Distributed Dedup:")
        print(f"    Dedup hits: {dedup_stats.dedup_hits}")
        print(f"    Bytes saved: {dedup_stats.dedup_bytes_saved / 1024**2:.1f} MB")
        print(f"  Novelty 3 — Locality-Aware Placement:")
        print(f"    Promotions: {locality_stats.promotions_to_local}")
        print(f"  Scaling:")
        print(f"    Write: {scaling['write_gbps']:.2f} GB/s")
        print(f"    Read:  {scaling['read_gbps']:.2f} GB/s")
        print(f"\n✅ All tests completed!")


def test_single_node_simulation(rank, world):
    """Fallback when MPI is not available — test with local CascadeStore."""
    print_rank0("\n  Using CascadeStore (single-node) for feature validation", rank)
    
    cfg = cascade_cpp.CascadeConfig()
    cfg.shm_capacity = 50 * 1024 * 1024  # 50 MB
    cfg.gpu_capacity = 10 * 1024 * 1024   # 10 MB
    cfg.kv_compression = True
    cfg.dedup_enabled = True

    store = cascade_cpp.CascadeStore(cfg)

    # Dedup test
    num_prefix = 20
    for i in range(num_prefix):
        bid = block_id(0, 0, i, prefix="shared")
        data = generate_kv_block(0, 0, i)
        store.put(bid, data, True)

    # Write same blocks again → dedup
    for i in range(num_prefix):
        bid = block_id(0, 0, i, prefix="shared")
        data = generate_kv_block(0, 0, i)
        store.put(bid, data, True)

    stats = store.get_stats()
    print(f"\n  Single-node dedup test:")
    print(f"    Dedup hits: {stats.dedup_hits}")
    print(f"    Compression savings: {stats.compression_savings_bytes / 1024:.1f} KB")

    # Eviction test
    block_size = 1024 * 1024  # 1MB
    for i in range(60):
        bid = f"evict_{i:04d}"
        data = np.random.randint(0, 255, block_size, dtype=np.uint8)
        store.put(bid, data, False)

    stats = store.get_stats()
    print(f"\n  Single-node eviction test:")
    print(f"    SHM used: {stats.shm_used_bytes / 1024**2:.1f} MB")
    print(f"    Evictions to Lustre: {stats.evictions_to_lustre}")

    print(f"\n✅ Single-node simulation complete!")


if __name__ == "__main__":
    main()
