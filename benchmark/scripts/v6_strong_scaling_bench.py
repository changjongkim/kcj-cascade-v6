#!/usr/bin/env python3
"""
Cascade V6 Strong Scaling Benchmark
------------------------------------
Strong Scaling: 전체 데이터 크기를 고정하고, 노드 수를 늘려서 처리 시간 단축을 측정.
- 전체 80,000 blocks (= ~12.5 GB uncompressed) 고정
- 노드 수: 1, 2, 4, 8 → rank당 할당량이 줄어듦
- 이상적 Strong Scaling: 노드 2배 → 시간 절반
"""

import sys
import os
import time
import numpy as np

# Path setup
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

def generate_block(seed, size=160*1024):
    np.random.seed(seed)
    return np.random.randint(0, 255, size, dtype=np.uint8)

def print_rank0(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)

def main():
    rank, world, node = get_rank_info()
    
    # ─── Configuration ───
    cfg = cascade_cpp.DistributedConfig()
    cfg.num_gpus_per_node = 4
    cfg.gpu_capacity_per_device = 512 * 1024**2   # 512MB per GPU → 2GB/node
    cfg.dram_capacity = 1024 * 1024**2            # 1GB DRAM per node
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.kv_compression = True
    cfg.promotion_threshold = 2
    
    lustre_path = os.environ.get('CASCADE_LUSTRE_PATH',
                                  f'/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_strong_r{rank}')
    os.makedirs(lustre_path, exist_ok=True)
    cfg.lustre_path = lustre_path
    
    store = cascade_cpp.DistributedStore(cfg)
    
    block_size = 160 * 1024  # 160KB per block
    
    # ========================================================================
    # Strong Scaling: 전체 블록 수 고정 = 80,000 개
    # ========================================================================
    TOTAL_BLOCKS = 80000
    blocks_per_rank = TOTAL_BLOCKS // world
    total_data_gb = (TOTAL_BLOCKS * block_size) / 1024**3
    
    print_rank0("=" * 80, rank)
    print_rank0(f" CASCADE V6 STRONG SCALING BENCHMARK (World: {world})", rank)
    print_rank0("=" * 80, rank)
    print_rank0(f" Total Blocks: {TOTAL_BLOCKS} ({total_data_gb:.2f} GB)", rank)
    print_rank0(f" Blocks/Rank:  {blocks_per_rank}", rank)
    print_rank0(f" GPU Cap: {cfg.gpu_capacity_per_device // (1024**2)}MB/device", rank)
    print_rank0(f" DRAM Cap: {cfg.dram_capacity // (1024**2)}MB/node", rank)
    print_rank0(f" Novelties: Dedup={cfg.dedup_enabled}, Semantic={cfg.semantic_eviction}, "
                f"Locality={cfg.locality_aware}", rank)
    print_rank0("=" * 80, rank)
    
    # ─── Phase 1: Shared Prefix Put ───
    num_prefix = 500
    prefix_keys = [f"prefix_b{i:06d}" for i in range(num_prefix)]
    
    t0 = time.time()
    for key in prefix_keys:
        data = generate_block(hash(key) % 100000, block_size)
        store.put(key, data, is_prefix=True)
    store.sync_metadata()
    store.barrier()
    t_phase1 = time.time() - t0
    
    stats = store.get_stats()
    print_rank0(f"[Phase 1] Shared Prefix Put: {t_phase1:.3f}s", rank)
    print_rank0(f"          Dedup Hits: {stats.dedup_hits} (saved {stats.dedup_bytes_saved/1024**2:.1f} MB)", rank)
    print_rank0(f"          Total Blocks: {stats.total_blocks}", rank)
    
    # ─── Phase 2: Strong Scaling Write (Fixed Total) ───
    # 각 rank는 자기 할당분만 씀
    my_start = rank * blocks_per_rank
    my_end = my_start + blocks_per_rank
    suffix_keys = [f"suffix_b{i:06d}" for i in range(my_start, my_end)]
    
    store.barrier()
    t0 = time.time()
    for key in suffix_keys:
        data = generate_block(hash(key) % 100000, block_size)
        store.put(key, data, is_prefix=False)
    store.barrier()
    t_phase2 = time.time() - t0
    
    store.sync_metadata()
    store.barrier()
    
    stats = store.get_stats()
    write_gbps = total_data_gb / t_phase2
    
    print_rank0(f"[Phase 2] Strong Scaling Write: {t_phase2:.3f}s", rank)
    print_rank0(f"          Total Data: {total_data_gb:.2f} GB (fixed)", rank)
    print_rank0(f"          Throughput: {write_gbps:.2f} GB/s", rank)
    print_rank0(f"          DRAM Evictions: {stats.dram_evictions}", rank)
    print_rank0(f"          Cluster GPU Used: {stats.cluster_gpu_used/1024**2:.1f} MB", rank)
    print_rank0(f"          Cluster DRAM Used: {stats.cluster_dram_used/1024**2:.1f} MB", rank)
    
    # ─── Phase 3: Strong Scaling Read (Fixed Total) ───
    # 각 rank가 자기 할당분을 읽음
    store.barrier()
    t0 = time.time()
    for key in suffix_keys:
        buf = np.empty(block_size, dtype=np.uint8)
        store.get(key, buf)
    store.barrier()
    t_phase3 = time.time() - t0
    
    read_gbps = total_data_gb / t_phase3
    stats = store.get_stats()
    
    print_rank0(f"[Phase 3] Strong Scaling Read: {t_phase3:.3f}s", rank)
    print_rank0(f"          Throughput: {read_gbps:.2f} GB/s", rank)
    print_rank0(f"          Local GPU Hits: {stats.local_gpu_hits}", rank)
    print_rank0(f"          Local DRAM Hits: {stats.local_dram_hits}", rank)
    print_rank0(f"          Lustre Hits: {stats.lustre_hits}", rank)
    
    # ─── Summary ───
    if rank == 0:
        print("\n" + "=" * 80)
        print(" STRONG SCALING RESULTS")
        print("=" * 80)
        print(f" Nodes:          {world // 4} (Ranks: {world})")
        print(f" Total Data:     {total_data_gb:.2f} GB (FIXED)")
        print(f" Write Time:     {t_phase2:.3f}s")
        print(f" Write Throughput: {write_gbps:.2f} GB/s")
        print(f" Read Time:      {t_phase3:.3f}s")
        print(f" Read Throughput: {read_gbps:.2f} GB/s")
        print(f" Ideal Speedup:  {world // 4}x")
        print("=" * 80)

if __name__ == "__main__":
    main()
