#!/usr/bin/env python3
"""
Cascade V6 Remote Read Verification & Benchmark
================================================
Verifies that Tier 3/4 (Remote GPU/DRAM read via RDMA) actually works.

Phase 1: Each rank writes unique blocks locally (~50GB total)
Phase 2: sync_metadata() — share block locations across all nodes
Phase 3: Local Read — verify own data integrity (baseline)
Phase 4: Remote Read — read OTHER ranks' data via RDMA, verify integrity
Phase 5: Stats — remote_gpu_hits / remote_dram_hits MUST be > 0
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
    print("ERROR: cascade_cpp not found. Build with USE_MPI=ON first.")
    sys.exit(1)

def get_rank_info():
    rank = int(os.environ.get('SLURM_PROCID', os.environ.get('PMI_RANK', 0)))
    world = int(os.environ.get('SLURM_NTASKS', os.environ.get('PMI_SIZE', 1)))
    node = os.environ.get('SLURMD_NODENAME', 'localhost')
    return rank, world, node

def generate_block_deterministic(global_block_id, size=160*1024):
    """Generate deterministic data from global block ID so any rank can verify."""
    np.random.seed(global_block_id % (2**31))
    data = np.random.randint(0, 255, size, dtype=np.uint8)
    # Embed the block ID in the first 4 bytes for extra safety
    data.view(np.uint32)[0] = global_block_id
    return data

def print_rank0(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)

def main():
    rank, world, node = get_rank_info()
    
    # ─── Configuration ───
    cfg = cascade_cpp.DistributedConfig()
    cfg.num_gpus_per_node = 4
    cfg.gpu_capacity_per_device = 2048 * 1024**2   # 2GB per GPU (8GB/node)
    cfg.dram_capacity = 16384 * 1024**2             # 16GB DRAM per node (large, avoid eviction)
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.kv_compression = False    # Disable lossy compression for bit-perfect verification
    cfg.promotion_threshold = 3
    
    lustre_path = os.environ.get('CASCADE_LUSTRE_PATH',
                                  f'/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_verify_r{rank}')
    os.makedirs(lustre_path, exist_ok=True)
    cfg.lustre_path = lustre_path
    
    store = cascade_cpp.DistributedStore(cfg)
    
    block_size = 160 * 1024  # 160KB
    BLOCKS_PER_RANK = 10000
    TOTAL_BLOCKS = BLOCKS_PER_RANK * world
    total_data_gb = (TOTAL_BLOCKS * block_size) / 1024**3
    
    print_rank0("=" * 80, rank)
    print_rank0(f" CASCADE V6 REMOTE READ VERIFICATION (World: {world})", rank)
    print_rank0("=" * 80, rank)
    print_rank0(f" Total Blocks:      {TOTAL_BLOCKS} ({total_data_gb:.2f} GB)", rank)
    print_rank0(f" Blocks/Rank:       {BLOCKS_PER_RANK}", rank)
    print_rank0(f" GPU Cap:           {cfg.gpu_capacity_per_device // (1024**2)}MB/device", rank)
    print_rank0(f" DRAM Cap:          {cfg.dram_capacity // (1024**2)}MB/node", rank)
    print_rank0(f" Novelties:         Dedup={cfg.dedup_enabled}, Semantic={cfg.semantic_eviction}, Locality={cfg.locality_aware}", rank)
    print_rank0("=" * 80, rank)
    
    # ========================================================================
    # Phase 1: Write — Each rank writes its own blocks (Local Write)
    # ========================================================================
    my_start = rank * BLOCKS_PER_RANK
    my_end = my_start + BLOCKS_PER_RANK
    my_keys = [f"v_r{rank:03d}_b{i:06d}" for i in range(BLOCKS_PER_RANK)]
    my_data_gb = (BLOCKS_PER_RANK * block_size) / 1024**3
    
    store.barrier()
    t0 = time.time()
    for i in range(BLOCKS_PER_RANK):
        global_id = my_start + i
        data = generate_block_deterministic(global_id, block_size)
        store.put(my_keys[i], data, is_prefix=False)
    store.barrier()
    t_write = time.time() - t0
    
    write_gbps = total_data_gb / t_write
    print_rank0(f"[Phase 1] Write:     {t_write:.3f}s ({write_gbps:.2f} GB/s)", rank)
    
    # ========================================================================
    # Phase 2: Sync Metadata — Share block locations across all nodes
    # ========================================================================
    store.sync_metadata()
    store.barrier()
    print_rank0("[Phase 2] Metadata synced across all nodes", rank)
    
    # ========================================================================
    # Phase 3: Local Read — Verify own data integrity
    # ========================================================================
    local_verified = 0
    local_errors = 0
    
    store.barrier()
    t0 = time.time()
    for i in range(BLOCKS_PER_RANK):
        global_id = my_start + i
        buf = np.empty(block_size, dtype=np.uint8)
        ok = store.get(my_keys[i], buf)
        
        if ok:
            expected = generate_block_deterministic(global_id, block_size)
            if np.array_equal(buf[:block_size], expected):
                local_verified += 1
            else:
                local_errors += 1
                if local_errors <= 3:  # Only print first 3 errors
                    print(f"[Rank {rank}] LOCAL VERIFY ERROR: key={my_keys[i]}, "
                          f"first_bytes_got={buf[:8].tolist()}, "
                          f"first_bytes_exp={expected[:8].tolist()}", flush=True)
        else:
            local_errors += 1
            if local_errors <= 3:
                print(f"[Rank {rank}] LOCAL GET FAILED: key={my_keys[i]}", flush=True)
    
    store.barrier()
    t_local_read = time.time() - t0
    
    local_read_gbps = total_data_gb / t_local_read
    print_rank0(f"[Phase 3] Local Read:  {t_local_read:.3f}s ({local_read_gbps:.2f} GB/s)", rank)
    print_rank0(f"          Verified: {local_verified}/{BLOCKS_PER_RANK} per rank", rank)
    if local_errors > 0:
        print(f"[Rank {rank}] ⚠️  LOCAL ERRORS: {local_errors}", flush=True)
    
    # ========================================================================
    # Phase 4: Remote Read — Read OTHER ranks' data (RDMA Verification)
    # ========================================================================
    # Each rank reads 5000 blocks from other ranks (round-robin)
    REMOTE_READS_PER_RANK = min(5000, BLOCKS_PER_RANK)
    remote_verified = 0
    remote_errors = 0
    remote_misses = 0
    
    store.barrier()
    t0 = time.time()
    
    for i in range(REMOTE_READS_PER_RANK):
        # Pick a different rank's block (round-robin across other ranks)
        target_rank = (rank + 1 + (i % (world - 1))) % world
        target_block_idx = i % BLOCKS_PER_RANK
        target_global_id = target_rank * BLOCKS_PER_RANK + target_block_idx
        target_key = f"v_r{target_rank:03d}_b{target_block_idx:06d}"
        
        buf = np.empty(block_size, dtype=np.uint8)
        ok = store.get(target_key, buf)
        
        if ok:
            expected = generate_block_deterministic(target_global_id, block_size)
            if np.array_equal(buf[:block_size], expected):
                remote_verified += 1
            else:
                remote_errors += 1
                if remote_errors <= 3:
                    print(f"[Rank {rank}] REMOTE VERIFY ERROR: key={target_key} from rank {target_rank}, "
                          f"first_bytes_got={buf[:8].tolist()}, "
                          f"first_bytes_exp={expected[:8].tolist()}", flush=True)
        else:
            remote_misses += 1
            if remote_misses <= 3:
                print(f"[Rank {rank}] REMOTE GET MISS: key={target_key} from rank {target_rank}", flush=True)
    
    store.barrier()
    t_remote_read = time.time() - t0
    
    remote_data_gb = (REMOTE_READS_PER_RANK * world * block_size) / 1024**3
    remote_read_gbps = remote_data_gb / t_remote_read if t_remote_read > 0 else 0
    
    print_rank0(f"[Phase 4] Remote Read: {t_remote_read:.3f}s ({remote_read_gbps:.2f} GB/s)", rank)
    
    # Collect verification counts (simple print per rank, rank 0 summarizes)
    if remote_errors > 0 or remote_misses > 0:
        print(f"[Rank {rank}] Remote: verified={remote_verified}, errors={remote_errors}, misses={remote_misses}", flush=True)
    
    # ========================================================================
    # Phase 5: Stats — CRITICAL: remote hits MUST be > 0
    # ========================================================================
    stats = store.get_stats()
    
    store.barrier()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(" VERIFICATION RESULTS")
        print("=" * 80)
        print(f" Nodes:              {world // 4}")
        print(f" Total Data:         {total_data_gb:.2f} GB")
        print(f" Write Throughput:   {write_gbps:.2f} GB/s")
        print(f" Local Read:         {local_read_gbps:.2f} GB/s")
        print(f" Remote Read:        {remote_read_gbps:.2f} GB/s")
        print(f"")
        print(f" ──── HIT STATISTICS (Rank 0) ────")
        print(f" Local GPU Hits:     {stats.local_gpu_hits}")
        print(f" Local DRAM Hits:    {stats.local_dram_hits}")
        print(f" Remote GPU Hits:    {stats.remote_gpu_hits}")
        print(f" Remote DRAM Hits:   {stats.remote_dram_hits}")
        print(f" Lustre Hits:        {stats.lustre_hits}")
        print(f" Misses:             {stats.misses}")
        print(f" Dedup Hits:         {stats.dedup_hits}")
        print(f"")
        print(f" Cluster GPU Used:   {stats.cluster_gpu_used / (1024**2):.1f} MB")
        print(f" Cluster DRAM Used:  {stats.cluster_dram_used / (1024**2):.1f} MB")
        print(f"")
        
        # CRITICAL VERIFICATION
        remote_total = stats.remote_gpu_hits + stats.remote_dram_hits
        if remote_total > 0:
            print(f" ✅ TIER 3/4 VERIFIED: {remote_total} remote hits detected!")
            print(f"    Remote GPU Hits:  {stats.remote_gpu_hits}")
            print(f"    Remote DRAM Hits: {stats.remote_dram_hits}")
        else:
            print(f" ❌ TIER 3/4 FAILED: 0 remote hits! RDMA not working!")
        
        local_ok = (local_verified == BLOCKS_PER_RANK)
        remote_ok = (remote_verified == REMOTE_READS_PER_RANK and remote_errors == 0 and remote_misses == 0)
        
        if local_ok and remote_ok:
            print(f" ✅ DATA INTEGRITY: All local + remote reads verified correctly!")
        else:
            if not local_ok:
                print(f" ❌ LOCAL INTEGRITY FAILED: {local_verified}/{BLOCKS_PER_RANK}")
            if not remote_ok:
                print(f" ❌ REMOTE INTEGRITY FAILED: verified={remote_verified}, "
                      f"errors={remote_errors}, misses={remote_misses}")
        
        print("=" * 80)

if __name__ == "__main__":
    main()
