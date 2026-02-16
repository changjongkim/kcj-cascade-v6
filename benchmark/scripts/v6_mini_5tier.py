#!/usr/bin/env python3
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

def main():
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world = int(os.environ.get('SLURM_NTASKS', 1))
    
    # ─── Configuration (Very Small Capacities to Force All Tiers) ───
    cfg = cascade_cpp.DistributedConfig()
    cfg.num_gpus_per_node = 4
    
    # 1. GPU 용량을 8MB로 극도로 제한 (노드당 32MB)
    cfg.gpu_capacity_per_device = 8 * 1024 * 1024 
    # 2. DRAM 용량을 16MB로 극도로 제한
    cfg.dram_capacity = 16 * 1024 * 1024
    
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.kv_compression = False # 검증을 위해 압축은 끔 (무결성 확인용)
    
    lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_mini_r{rank}"
    os.makedirs(lustre_path, exist_ok=True)
    cfg.lustre_path = lustre_path
    
    store = cascade_cpp.DistributedStore(cfg)
    block_size = 160 * 1024 # 160KB
    
    # 노드당 400MB를 써서 Lustre까지 넘치게 함 (노드당 4 rank이므로 rank당 100MB)
    BLOCKS_PER_RANK = 640 # 100MB / 160KB
    TOTAL_BLOCKS = BLOCKS_PER_RANK * world
    
    if rank == 0:
        print("=" * 80)
        print(f" 🔍 CASCADE V6 - MINI 5-TIER VERIFICATION")
        print(f" Total Capacity: GPU=128MB/node, DRAM=64MB/node (Total 192MB)")
        print(f" Target Writing: 400MB/node (Should trigger Lustre)")
        print("=" * 80)

    # ─── Phase 1: Write (Tier 1, 2, 5) ───
    sample_data = np.random.randint(0, 255, block_size, dtype=np.uint8)
    data_view = sample_data.view(np.uint32)
    
    store.barrier()
    for i in range(BLOCKS_PER_RANK):
        # 모든 블록이 유니크하도록 데이터 수정
        data_view[0] = i + (rank * 100000)
        store.put(f"mini_r{rank:03d}_b{i:04d}", sample_data)
        
    store.barrier()
    store.sync_metadata()
    store.barrier()

    # ─── Phase 2: Random Read (Tier 3, 4, 5) ───
    # 각 Rank가 전체 키 공간에서 무작위로 200번 읽음
    READ_SAMPLES = 200
    np.random.seed(rank + 7)
    read_ranks = np.random.randint(0, world, READ_SAMPLES)
    read_blocks = np.random.randint(0, BLOCKS_PER_RANK, READ_SAMPLES)
    
    store.barrier()
    for i in range(READ_SAMPLES):
        r = read_ranks[i]
        b = read_blocks[i]
        buf = np.empty(block_size, dtype=np.uint8)
        store.get(f"mini_r{r:03d}_b{b:04d}", buf)
        
    store.barrier()
    stats = store.get_stats()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(" [MINI 5-TIER HIT STATISTICS]")
        print("=" * 80)
        print(f" Tier 1 (Local GPU):   {stats.local_gpu_hits}")
        print(f" Tier 2 (Local DRAM):  {stats.local_dram_hits}")
        print(f" Tier 3 (Remote GPU):  {stats.remote_gpu_hits}")
        print(f" Tier 4 (Remote DRAM): {stats.remote_dram_hits}")
        print(f" Tier 5 (Lustre):      {stats.lustre_hits}")
        print("-" * 80)
        
        # 모든 티어가 작동했는지 자동 검사
        tiers = {
            "Local GPU": stats.local_gpu_hits,
            "Local DRAM": stats.local_dram_hits,
            "Remote GPU": stats.remote_gpu_hits,
            "Remote DRAM": stats.remote_dram_hits,
            "Lustre": stats.lustre_hits
        }
        
        working = [t for t, h in tiers.items() if h > 0]
        failed = [t for t, h in tiers.items() if h == 0]
        
        if not failed:
            print(" ✅ SUCCESS: All 5 tiers are working correctly!")
        else:
            print(f" ⚠️  PARTIAL: Working: {working}")
            print(f" ❌ FAILED: Zero hits for: {failed}")
        print("=" * 80)

if __name__ == "__main__":
    main()
