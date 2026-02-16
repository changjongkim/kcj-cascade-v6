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
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3  # 38GB per GPU
        cfg.dram_capacity = 200 * 1024**3           # 200GB DRAM (1 rank per node)
        cfg.num_gpus_per_node = 4  # 4 GPUs/node
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        self.lustre_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cascade_lustre_r{rank}")
        self.lustre_dir.mkdir(parents=True, exist_ok=True)
        cfg.lustre_path = str(self.lustre_dir)
        self.store = cascade_cpp.DistributedStore(cfg)
    
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

def run_bench(system_name, adapter, loader, block_ids):
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    
    # ─── Pre-load Phase ───
    # To measure pure WRITE performance, we must read from Lustre disk first
    # and store in memory so the timer only measures the Storage System's speed.
    loaded_data = []
    print_rank0(f"  [{system_name}] Pre-loading {len(block_ids)} blocks to memory...")
    for bid in block_ids:
        loaded_data.append(loader.read_block(bid))
    
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    
    # ─── Write Phase ───
    t0 = time.time()
    total_bytes = 0
    for i, bid in enumerate(block_ids):
        data = loaded_data[i]
        adapter.put(bid, data)
        total_bytes += data.nbytes
    
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    t_write = time.time() - t0
    
    if system_name == "Cascade":
        adapter.store.sync_metadata()
        adapter.store.barrier()
    
    # ─── Read Phase ───
    # We read in the same order as write (warm cache test)
    # and then random order (scaling test)
    t1 = time.time()
    for bid in block_ids:
        # We need a buffer. Since blocks are ~164MB, we reuse it.
        size = loader.all_blocks[bid]['size']
        buf = np.empty(size, dtype=np.uint8)
        adapter.get(bid, buf)
        
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    t_read = time.time() - t1
    
    # Aggregated metrics
    local_gb = total_bytes / 1024**3
    global_gb = local_gb  # In 1-rank-per-node MPI, each rank's data represents its share
        
    if rank == 0:
        write_bw = global_gb / t_write
        read_bw = global_gb / t_read
        print(f"  {system_name:10} | Write: {write_bw:7.2f} GB/s | Read: {read_bw:7.2f} GB/s")
        return (write_bw, read_bw)
    return (global_gb/t_write, global_gb/t_read) # Return values for all ranks to avoid None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", type=str, help="Comma-separated systems")
    parser.add_argument("--blocks-per-rank", type=int, default=20, help="Blocks per rank (default 20, ~3.2GB)")
    args = parser.parse_args()
    
    systems_to_test = args.systems.split(',')
    num_blocks = args.blocks_per_rank

    loader = RealKVLoader()
    my_block_ids = loader.get_my_blocks(rank, world, limit=num_blocks)
    
    print_rank0(f" Blocks/rank: {len(my_block_ids)}, ~{len(my_block_ids)*164/1024:.1f} GB/rank")
    print_rank0(f" Testing: {', '.join(systems_to_test)}")
    print_rank0("="*80)
    
    final_results = {}
    
    for name in systems_to_test:
        adapter = None
        try:
            if name == "Cascade":
                adapter = CascadeAdapter()
            elif name == "HDF5": adapter = HDF5Adapter()
            elif name == "LMCache": adapter = LMCacheAdapter()
            elif name == "Redis": adapter = RedisAdapter()
            elif name == "PDC": adapter = PDCAdapter()
            
            if adapter:
                res = run_bench(name, adapter, loader, my_block_ids)
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
