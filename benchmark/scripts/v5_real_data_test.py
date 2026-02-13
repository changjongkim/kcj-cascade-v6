import os
import sys
import json
import time
import struct
import hashlib
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add Cascade paths
REPO_ROOT = Path("/pscratch/sd/s/sgkim/kcj/Cascade-kcj")
CPP_BUILD = REPO_ROOT / "cascade_Code/cpp/build"
sys.path.insert(0, str(CPP_BUILD))
sys.path.insert(0, str(REPO_ROOT))

import cascade_cpp

# Aggregated data directory
DATA_DIR = Path("/pscratch/sd/s/sgkim/cascade_kv_cache")

class RealDataReader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        index_path = data_dir / "global_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        with open(index_path) as f:
            self.index = json.load(f).get("blocks", {})
    
    def get_block_ids(self):
        return list(self.index.keys())
    
    def read_block(self, block_id: str):
        info = self.index[block_id]
        with open(self.data_dir / info["file"], "rb") as f:
            f.seek(info["offset"])
            # Format: <QQ (key_size, value_size)
            header = f.read(16)
            key_size, value_size = struct.unpack("<QQ", header)
            data = f.read(key_size + value_size)
            return np.frombuffer(data, dtype=np.uint8)

def run_real_data_bench():
    print("=" * 70)
    print("🚀 Cascade V5 Real Application Data Benchmark")
    print("=" * 70)
    
    # 1. Load data
    print(f"Loading metadata from {DATA_DIR}...")
    reader = RealDataReader(DATA_DIR)
    block_ids = reader.get_block_ids()
    print(f"Total blocks available in trace: {len(block_ids)}")
    
    NUM_TEST_BLOCKS = 100 # ~16GB
    test_block_ids = block_ids[:NUM_TEST_BLOCKS]
    
    print(f"Reading {NUM_TEST_BLOCKS} real blocks into memory...")
    blocks = []
    total_size_bytes = 0
    t0 = time.time()
    for bid in test_block_ids:
        data = reader.read_block(bid)
        if data is not None:
            blocks.append((bid, data))
            total_size_bytes += len(data)
    t1 = time.time()
    
    total_gb = total_size_bytes / (1024**3)
    print(f"Loaded {len(blocks)} blocks ({total_gb:.2f} GB) in {t1-t0:.2f}s")
    
    # 2. Configure Cascade Store (V5)
    config = cascade_cpp.CascadeConfig()
    config.use_gpu = False # Testing SHM/Lustre first, or set True if on GPU node
    config.shm_capacity_bytes = 8 * 1024 * 1024 * 1024  # 8GB SHM (force some evictions)
    config.lustre_path = "/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/data/real_lustre_test"
    config.semantic_eviction = True
    
    # Ensure lustre path is clean
    if os.path.exists(config.lustre_path):
        import shutil
        shutil.rmtree(config.lustre_path)
    os.makedirs(config.lustre_path, exist_ok=True)
    
    store = cascade_cpp.CascadeStore(config)
    print(f"\nInitialized Cascade Store (SHM={config.shm_capacity_bytes/(1024**3):.1f}GB, Lustre={config.lustre_path})")
    
    # 3. Write Test
    print(f"\n[1/3] Performance Write ({len(blocks)} blocks)...")
    start = time.time()
    for bid, data in blocks:
        store.put(bid, data, False)
    end = time.time()
    
    duration = end - start
    throughput = total_gb / duration
    print(f"      Throughput: {throughput:.2f} GB/s")
    
    stats = store.get_stats()
    print(f"      Stats: SHM used={stats.shm_used/(1024**2):.1f}MB, Evictions={stats.shm_evictions}")
    
    print(f"\n[2/3] Performance Read ({len(blocks)} blocks)...")
    out_buf = np.zeros(blocks[0][1].size, dtype=np.uint8)
    start = time.time()
    for bid, _ in blocks:
        found, size = store.get(bid, out_buf)
    end = time.time()
    
    duration = end - start
    throughput = total_gb / duration
    print(f"      Throughput: {throughput:.2f} GB/s")
    print(f"      Hits: SHM={store.get_stats().shm_hits}, Lustre={store.get_stats().lustre_hits}")
    
    # 5. Dedup Test (Shared Prefix Simulation)
    print(f"\n[3/3] Deduplication Test (Re-writing same blocks)...")
    store.clear()
    
    # Simulate prefix: first 5 blocks are prefixes
    prefixes = blocks[:10]
    unique = blocks[10:30]
    
    # 5 users sharing 10 prefixes, but having different uniques
    all_reqs = []
    for user in range(5):
        for bid, data in prefixes:
            all_reqs.append((bid, data, True)) # is_prefix=True, use same bid for dedup
        for bid, data in unique:
            all_reqs.append((f"user{user}_u_{bid}", data, False))
            
    total_dedup_bytes = sum(len(d) for _, d, _ in all_reqs)
    total_dedup_gb = total_dedup_bytes / (1024**3)
    
    start = time.time()
    for bid, data, is_pref in all_reqs:
        store.put(bid, data, is_pref)
    end = time.time()
    
    duration = end - start
    throughput = total_dedup_gb / duration
    
    stats = store.get_stats()
    print(f"      Throughput: {throughput:.2f} GB/s (Includes dedup hits)")
    print(f"      Dedup Hits: {stats.dedup_hits}")
    print(f"      Total puts: {len(all_reqs)}, Unique writes: {stats.shm_puts + stats.lustre_puts}")
    
    print("\n" + "=" * 70)
    print("✅ Real App Data Experiment Complete!")
    print("=" * 70)

if __name__ == "__main__":
    run_real_data_bench()
