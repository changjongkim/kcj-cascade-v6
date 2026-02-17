#!/usr/bin/env python3
"""
Cascade V6 Real-Data Comparison Benchmark
=========================================
Compares Cascade vs HDF5 vs PDC using pre-generated 500GB real KV cache data.
Data location: /pscratch/sd/s/sgkim/cascade_kv_cache/
Blocks: ~164MB each (key+value aggregated)
"""

import os
import sys
import time
import json
import numpy as np
import struct
from pathlib import Path

# MPI Configuration: Use SLURM environment variables to avoid mpi4py double-init
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
comm = None

# Path setup for Cascade library
default_build = '../../cascade_Code/cpp/build_cascade_cpp'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), build_dir)
sys.path.insert(0, build_dir)

try:
    import cascade_cpp
except ImportError:
    cascade_cpp = None

def print_rank0(msg):
    if rank == 0:
        print(msg, flush=True)

# ============================================================
# Data Loader for Aggregated Files
# ============================================================

class RealKVLoader:
    def __init__(self, base_dir="/pscratch/sd/s/sgkim/cascade_kv_cache"):
        self.base_dir = Path(base_dir)
        index_path = self.base_dir / "global_index.json"
        with open(index_path, 'r') as f:
            data = json.load(f)
            self.all_blocks = data['blocks']
            self.block_ids = list(self.all_blocks.keys())
            
    def get_my_blocks(self, rank, world, limit=50):
        """Assign blocks to this rank."""
        # Distribute unique blocks across ranks
        my_ids = self.block_ids[rank::world]
        if limit and len(my_ids) > limit:
            my_ids = my_ids[:limit]
        return my_ids

    def read_blocks(self, block_ids):
        """Read multiple blocks efficiently by grouping by file."""
        from collections import defaultdict
        file_to_blocks = defaultdict(list)
        for bid in block_ids:
            loc = self.all_blocks[bid]
            file_to_blocks[loc['file']].append(bid)
        
        results = {}
        total = len(block_ids)
        current = 0
        
        for rel_path, bids in file_to_blocks.items():
            abs_path = self.base_dir / rel_path
            with open(abs_path, 'rb') as f:
                for bid in bids:
                    loc = self.all_blocks[bid]
                    f.seek(loc['offset'])
                    results[bid] = np.frombuffer(f.read(loc['size']), dtype=np.uint8)
                    current += 1
                    if current % 10 == 0 or current == total:
                        print_rank0(f"    - Loaded {current}/{total} blocks...")
        return results

    def read_block(self, block_id):
        """Read actual data from aggregated file."""
        if block_id not in self.all_blocks:
            raise KeyError(f"Block ID {block_id} not found in index! (Dict size: {len(self.all_blocks)})")
        loc = self.all_blocks[block_id] # Take location info
        path = self.base_dir / loc['file']
        offset = loc['offset']
        size = loc['size']
        
        with open(path, 'rb') as f:
            f.seek(offset)
            return np.frombuffer(f.read(size), dtype=np.uint8)

# ============================================================
# Storage Adapters
# ============================================================

class BaseStore:
    def put(self, key, data): pass
    def get(self, key, out_size): pass
    def cleanup(self): pass

class CascadeAdapter(BaseStore):
    def __init__(self):
        print_rank0(f"  [Cascade] Initializing DistributedStore (128GB DRAM + 38GB/GPU)...")
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3  # 38GB per GPU
        cfg.dram_capacity = 128 * 1024**3           # 128GB DRAM (Balanced for 256GB total)
        cfg.num_gpus_per_node = 4  # 4 GPUs/node
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        self.lustre_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cascade_lustre_r{rank}")
        self.lustre_dir.mkdir(parents=True, exist_ok=True)
        cfg.lustre_path = str(self.lustre_dir)
        self.store = cascade_cpp.DistributedStore(cfg)
        print_rank0(f"  [Cascade] Store initialized.")
    
    def put(self, key, data):
        return self.store.put(key, data)
    
    def get(self, key, out):
        return self.store.get(key, out)

    def cleanup(self):
        import shutil
        if self.lustre_dir.exists():
            shutil.rmtree(self.lustre_dir)

class HDF5Adapter(BaseStore):
    def __init__(self):
        import h5py
        self.path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_comp_r{rank}.h5"
        self.file = h5py.File(self.path, 'w')
    
    def put(self, key, data):
        if key in self.file: del self.file[key]
        self.file.create_dataset(key, data=data)
        self.file.flush()
    
    def get(self, key, out):
        if key in self.file:
            dset = self.file[key]
            dset.read_direct(out)
            return True, dset.size
        return False, 0
    
    def cleanup(self):
        self.file.close()
        if os.path.exists(self.path): os.remove(self.path)

class LMCacheAdapter(BaseStore):
    def __init__(self):
        # Simulated as file-based (local per-rank scratch)
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lmcache_comp_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def put(self, key, data):
        with open(self.dir / f"{key}.bin", 'wb') as f:
            f.write(data)
    
    def get(self, key, out):
        p = self.dir / f"{key}.bin"
        if p.exists():
            with open(p, 'rb') as f:
                content = f.read()
                out[:len(content)] = np.frombuffer(content, dtype=np.uint8)
                return True, len(content)
        return False, 0
    
    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

class RedisAdapter(BaseStore):
    def __init__(self):
        import redis
        # Assumes redis-server is running on localhost:16379 (standard for our slurm scripts)
        self.client = redis.Redis(host='localhost', port=16379)
    
    def put(self, key, data):
        self.client.set(key, data.tobytes())
    
    def get(self, key, out):
        val = self.client.get(key)
        if val:
            out[:len(val)] = np.frombuffer(val, dtype=np.uint8)
            return True, len(val)
        return False, 0
    
    def cleanup(self):
        try:
            self.client.flushdb()
        except:
            pass

class PDCAdapter(BaseStore):
    """Simulates PDC/Direct Lustre with O_DIRECT if possible, or just fsync."""
    def __init__(self):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/pdc_comp_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def put(self, key, data):
        path = self.dir / f"{key}.bin"
        with open(path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    
    def get(self, key, out):
        p = self.dir / f"{key}.bin"
        if p.exists():
            with open(p, 'rb') as f:
                data = f.read()
                out[:len(data)] = np.frombuffer(data, dtype=np.uint8)
                return True, len(data)
        return False, 0
    
    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

# ============================================================
# Execution Loop
# ============================================================

def run_contention_bench(system_name, adapter, write_ids, read_ids, loaded_data_dict):
    """
    Contention Scenario:
    1. Write Unique blocks (to populate storage)
    2. Read SAME blocks across all ranks (Contention on the 'Hot Prefix')
    """
    write_bytes = sum(loaded_data_dict[bid].nbytes for bid in write_ids)
    read_bytes = sum(loaded_data_dict[bid].nbytes for bid in read_ids)
    
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    
    # ─── Phase 1: Unique Write (Preparation) ───
    t0 = time.time()
    for bid in write_ids:
        adapter.put(bid, loaded_data_dict[bid])
    
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    t_write = time.time() - t0
    
    if system_name == "Cascade":
        adapter.store.sync_metadata()
        adapter.store.barrier()
    
    # ─── Phase 2: Contended Read (The 'Hot Prefix' Test) ───
    # All ranks read EXACTLY the same read_ids at the same time
    print_rank0(f"  [{system_name}] Starting contention read of {len(read_ids)} shared blocks...")
    
    # Special case: For HDF5/LMCache, data must exist in the shared/local path for all ranks
    # In our current adapter, HDF5 uses local file per rank. To simulate contention,
    # we would need a single shared HDF5 file. (For now, we use the fact that they
    # all hit the same keys if shared storage was used).
    
    t1 = time.time()
    for bid in read_ids:
        size = loaded_data_dict[bid].nbytes
        buf = np.empty(size, dtype=np.uint8)
        adapter.get(bid, buf)
    
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    t_read = time.time() - t1
    
    # Metrics
    local_write_gb = write_bytes / 1024**3
    local_read_gb = read_bytes / 1024**3
        
    if rank == 0:
        write_bw = local_write_gb / t_write
        read_bw = local_read_gb / t_read
        print(f"  {system_name:10} | Unique Write: {write_bw:7.2f} GB/s | Contended Read: {read_bw:7.2f} GB/s", flush=True)
        return (write_bw, read_bw)
    return (local_write_gb/t_write, local_read_gb/t_read)
 # Return values for all ranks to avoid None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", type=str, help="Comma-separated systems")
    parser.add_argument("--blocks-per-rank", type=int, default=160, help="Blocks per rank (default 160, ~26GB)")
    args = parser.parse_args()
    
    systems_to_test = args.systems.split(',')
    num_blocks = args.blocks_per_rank

    loader = RealKVLoader()
    # Total blocks available for this test
    total_workload = num_blocks 
    num_hot = num_blocks // 4  # 25% is hot
    num_unique = num_blocks - num_hot
    
    # 1. Unique blocks for each rank to write
    write_ids = loader.get_my_blocks(rank, world, limit=num_unique)
    # 2. Hot blocks (same for all ranks)
    read_ids = loader.block_ids[:num_hot]
    
    all_needed = list(set(write_ids + read_ids))
    print_rank0(f" Contention Config: {num_unique} unique writes, {num_hot} shared reads")
    print_rank0(f" [Global] Pre-loading {len(all_needed)} blocks to memory...")
    loaded_data_dict = loader.read_blocks(all_needed)
    
    import gc
    gc.collect()
    
    print_rank0(f" Testing Contention: {', '.join(systems_to_test)}")
    print_rank0("="*80)
    print_rank0(f"{'System':12} | {'Write BW':>15} | {'Contended Read':>15}")
    
    final_results = {}
    
    for name in systems_to_test:
        adapter = None
        try:
            if name == "Cascade":
                adapter = CascadeAdapter()
            elif name == "HDF5": 
                adapter = HDF5Adapter()
                # For HDF5 contention read to work in this script, all ranks need the hot data
                # We simulate this by having all ranks write the hot blocks first if they don't have it
                for bid in read_ids:
                    if bid not in write_ids:
                        adapter.put(bid, loaded_data_dict[bid])
            elif name == "LMCache": 
                adapter = LMCacheAdapter()
                for bid in read_ids:
                    if bid not in write_ids:
                        adapter.put(bid, loaded_data_dict[bid])
            
            if adapter:
                res = run_contention_bench(name, adapter, write_ids, read_ids, loaded_data_dict)
                if rank == 0 and res: final_results[name] = res
                adapter.cleanup()
        except Exception as e:
            print(f"Error testing {name} on rank {rank}: {e}")
            import traceback
            traceback.print_exc()

    if rank == 0:
        print("\n" + "="*80)
        print(f" Final Comparison Summary ({world} Nodes)")
        print("="*80)
        print(f"{'System':12} | {'Write (GB/s)':>15} | {'Read (GB/s)':>15}")
        print("-" * 50)
        for name, bw in final_results.items():
            print(f"{name:12} | {bw[0]:15.2f} | {bw[1]:15.2f}")
        print("="*80)

if __name__ == "__main__":
    main()
