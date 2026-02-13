import os
import sys
import time
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))

try:
    from cascade_cpp import DistributedStore, DistributedConfig
    print(f"✅ cascade_cpp imported")
except ImportError as e:
    print(f"❌ Failed to import cascade_cpp: {e}")
    sys.exit(1)

def run_bench():
    # Configuration
    cfg = DistributedConfig()
    cfg.gpu_capacity_per_device = 1 * 1024 * 1024 * 1024  # 1GB
    cfg.dram_capacity = 2 * 1024 * 1024 * 1024           # 2GB
    cfg.num_gpus_per_node = 1
    
    # DistributedStore will call MPI_Init in C++ if not initialized
    store = DistributedStore(cfg)
    
    rank = store.rank
    size = store.world_size
    
    print(f"Rank {rank}/{size} starting (C++ MPI initialized)...")
    
    if size < 2:
        print("Need at least 2 nodes for multi-node benchmark")
        return

    # 100MB block
    block_size = 100 * 1024 * 1024
    np.random.seed(rank)
    data = np.random.randint(0, 256, block_size, dtype=np.uint8)
    block_id = f"rank_{rank}_block_0"
    
    # --- PHASE 1: WRITE ---
    print(f"[Rank {rank}] Writing {block_size/1e6:.1f} MB block...")
    t0 = time.perf_counter()
    store.put(block_id, data)
    t1 = time.perf_counter()
    if rank == 0:
        print(f"[Rank 0] Write done in {t1-t0:.3f}s")
    
    store.barrier()
    
    # --- PHASE 2: REMOTE READ ---
    peer_rank = (rank + 1) % size
    peer_block_id = f"rank_{peer_rank}_block_0"
    
    print(f"[Rank {rank}] Reading remote block {peer_block_id} from Rank {peer_rank}...")
    out = np.zeros(block_size, dtype=np.uint8)
    
    t0 = time.perf_counter()
    found, read_size = store.get(peer_block_id, out)
    t1 = time.perf_counter()
    
    if found:
        # Verification
        np.random.seed(peer_rank)
        expected = np.random.randint(0, 256, block_size, dtype=np.uint8)
        if np.array_equal(out, expected):
            print(f"[Rank {rank}] ✅ Remote read verified! {read_size/1e9/(t1-t0):.2f} GB/s")
        else:
            print(f"[Rank {rank}] ❌ Remote read DATA CORRUPT")
    else:
        print(f"[Rank {rank}] ❌ Remote block NOT found")
    
    store.barrier()
    
    stats = store.get_stats()
    if rank == 0:
        print("\n--- Distributed Stats ---")
        print(f"Local GPU/DRAM Hits: {stats.local_gpu_hits} / {stats.local_dram_hits}")
        print(f"Remote GPU/DRAM Hits: {stats.remote_gpu_hits} / {stats.remote_dram_hits}")
        print(f"Misses: {stats.misses}")

if __name__ == "__main__":
    run_bench()
