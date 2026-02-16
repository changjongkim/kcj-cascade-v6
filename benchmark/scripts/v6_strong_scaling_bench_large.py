#!/usr/bin/env python3
"""
Cascade V6 Strong Scaling Benchmark (Large Scale: 500GB)
------------------------------------
Strong Scaling: 전체 데이터 크기 고정 (500GB), 노드 수 증가 시 처리 시간 단축 측정.
- 전체 3,200,000 blocks (= ~500 GB uncompressed @ 160KB/block) 고정
- 노드 수: 1, 2, 4, 8
- 목표: 대규모 데이터셋(Lustre 병목 발생)에 대한 Cascade의 Tiering 효율 검증.
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
    cfg.gpu_capacity_per_device = 512 * 1024**2   # 512MB per GPU
    cfg.dram_capacity = 1024 * 1024**2            # 1GB DRAM per node (Small for testing eviction)
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.kv_compression = True
    cfg.promotion_threshold = 2
    
    lustre_path = os.environ.get('CASCADE_LUSTRE_PATH',
                                  f'/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_strong_large_r{rank}')
    os.makedirs(lustre_path, exist_ok=True)
    cfg.lustre_path = lustre_path
    
    store = cascade_cpp.DistributedStore(cfg)
    
    block_size = 160 * 1024  # 160KB per block
    
    # ========================================================================
    # Strong Scaling (Large): 3,200,000 blocks ≈ 500 GB
    # ========================================================================
    TOTAL_BLOCKS = 3200000 
    blocks_per_rank = TOTAL_BLOCKS // world
    total_data_gb = (TOTAL_BLOCKS * block_size) / 1024**3
    
    print_rank0("=" * 80, rank)
    print_rank0(f" CASCADE V6 STRONG SCALING [LARGE] (World: {world})", rank)
    print_rank0("=" * 80, rank)
    print_rank0(f" Total Blocks: {TOTAL_BLOCKS} ({total_data_gb:.2f} GB)", rank)
    print_rank0(f" Blocks/Rank:  {blocks_per_rank}", rank)
    print_rank0(f" GPU Cap: {cfg.gpu_capacity_per_device // (1024**2)}MB/device", rank)
    print_rank0(f" DRAM Cap: {cfg.dram_capacity // (1024**2)}MB/node", rank)
    print_rank0("=" * 80, rank)
    
    # ─── Phase 1: Warming Up (Prefix) ───
    # Skipping prefix heavy load to focus on raw throughput
    
    # ─── Phase 2: Strong Scaling Write (Fixed Total 500GB) ───
    my_start = rank * blocks_per_rank
    my_end = my_start + blocks_per_rank
    suffix_keys = [f"large_b{i:08d}" for i in range(my_start, my_end)]
    
    # Generate one block and create a view for fast modification
    sample_data = generate_block(rank, block_size)
    # Create a 32-bit int view of the first 4 bytes to inject unique ID
    # This ensures every block has a unique SHA256 hash, bypassing Dedup.
    # (We want to stress test actual I/O capacity, not Dedup efficiency here)
    data_view = sample_data.view(np.uint32)
    
    store.barrier()
    t0 = time.time()
    for i, key in enumerate(suffix_keys):
        # Inject unique index into the first 4 bytes
        data_view[0] = i + (rank * 10000000) 
        store.put(key, sample_data, is_prefix=False)

        
    store.barrier()
    t_phase2 = time.time() - t0
    
    store.sync_metadata()
    store.barrier()
    
    write_gbps = total_data_gb / t_phase2
    
    print_rank0(f"[Phase 2] Strong Scaling Write: {t_phase2:.3f}s", rank)
    print_rank0(f"          Total Data: {total_data_gb:.2f} GB (fixed)", rank)
    print_rank0(f"          Throughput: {write_gbps:.2f} GB/s", rank)
    
    # ─── Phase 3: Strong Scaling Read (Fixed Total 500GB) ───
    store.barrier()
    t0 = time.time()
    for key in suffix_keys:
        buf = np.empty(block_size, dtype=np.uint8)
        store.get(key, buf)
    store.barrier()
    t_phase3 = time.time() - t0
    
    read_gbps = total_data_gb / t_phase3
    
    print_rank0(f"[Phase 3] Strong Scaling Read: {t_phase3:.3f}s", rank)
    print_rank0(f"          Throughput: {read_gbps:.2f} GB/s", rank)
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(" STRONG SCALING [LARGE 500GB] RESULTS")
        print("=" * 80)
        print(f" Nodes:          {world // 4}")
        print(f" Total Data:     {total_data_gb:.2f} GB")
        print(f" Write Time:     {t_phase2:.3f}s ({write_gbps:.2f} GB/s)")
        print(f" Read Time:      {t_phase3:.3f}s ({read_gbps:.2f} GB/s)")
        print("=" * 80)

if __name__ == "__main__":
    main()
