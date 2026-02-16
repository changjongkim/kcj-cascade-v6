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
    def __init__(self, shared=False):
        import h5py
        # Shared file for contention test, local file for unique write
        self.path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_shared.h5" if shared else \
                    f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_comp_r{rank}.h5"
        self.file = h5py.File(self.path, 'a') # 'a' to allow multi-rank access
    
    def put(self, key, data):
        if key in self.file: return
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
        # Only rank0 remove shared file
        if "shared" in str(self.path) and rank == 0:
            if os.path.exists(self.path): os.remove(self.path)
        elif "shared" not in str(self.path):
            if os.path.exists(self.path): os.remove(self.path)

class LMCacheAdapter(BaseStore):
    def __init__(self, shared=False):
        # Simulated as file-based (SHARED path for contention)
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lmcache_shared") if shared else \
                   Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lmcache_comp_r{rank}")
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

class vLLMGPUAdapter(BaseStore):
    """Simulates vLLM disk swap (torch.save/load style)"""
    def __init__(self, shared=False):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/vllm_swap_shared") if shared else \
                   Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/vllm_swap_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def put(self, key, data):
        path = self.dir / f"{key}.pt"
        if not path.exists():
            with open(path, 'wb') as f:
                f.write(data)

    def get(self, key, out):
        p = self.dir / f"{key}.pt"
        if p.exists():
            with open(p, 'rb') as f:
                content = f.read()
                out[:len(content)] = np.frombuffer(content, dtype=np.uint8)
                return True, len(content)
        return False, 0
    
    def cleanup(self):
        import shutil
        if "shared" not in str(self.dir):
            if self.dir.exists(): shutil.rmtree(self.dir)
        elif rank == 0:
            if self.dir.exists(): shutil.rmtree(self.dir)

class PDCAdapter(BaseStore):
    """Simulates PDC (Parallel Data Component) / Lustre optimized access"""
    def __init__(self, shared=False):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/pdc_shared") if shared else \
                   Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/pdc_comp_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def put(self, key, data):
        path = self.dir / f"{key}.bin"
        if not path.exists():
            with open(path, 'wb') as f:
                f.write(data)
                f.flush()
                # os.fsync(f.fileno()) # Real-world simulation of persistence

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
        if "shared" not in str(self.dir):
            if self.dir.exists(): shutil.rmtree(self.dir)
        elif rank == 0:
            if self.dir.exists(): shutil.rmtree(self.dir)

# ============================================================
# Execution Loop
# ============================================================

def run_tiered_bench(system_name, adapter, write_ids, read_ids, loaded_data_dict):
    """
    Tiered Simulation:
    - Tier 1: GPU VRAM (40% hit rate, simulated 400 GB/s)
    - Tier 2: Storage Backend (Cascade/LMCache/HDF5)
    """
    num_total = len(read_ids)
    num_hits = int(0.4 * num_total) # 40% Hit Rate
    hit_ids = read_ids[:num_hits]
    miss_ids = read_ids[num_hits:]
    
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    
    # Prep: Write ALL read_ids so they are available in backend
    for bid in read_ids:
        adapter.put(bid, loaded_data_dict[bid])
    
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
        if system_name == "Cascade":
            adapter.store.sync_metadata()
            adapter.store.barrier()

    # ─── Read Phase (Tiered) ───
    t_start = time.time()
    
    # 1. GPU Hits (Simulated)
    # 400 GB/s is ~0.0004s per 164MB block
    hit_time = num_hits * (164 * 1024**2 / (400 * 1024**3))
    time.sleep(hit_time) # Mock GPU latency
    
    # 2. Storage Misses (Actual Backend Read with Contention)
    t_miss_start = time.time()
    for bid in miss_ids:
        size = loaded_data_dict[bid].nbytes
        buf = np.empty(size, dtype=np.uint8)
        adapter.get(bid, buf)
    
    if hasattr(adapter, 'store'):
        adapter.store.barrier()
    t_total = (time.time() - t_miss_start) + hit_time
    
    # ─── Metrics Calculation ───
    total_gb = sum(loaded_data_dict[bid].nbytes for bid in read_ids) / 1024**3
    throughput = total_gb / t_total
    
    if rank == 0:
        print(f"  {system_name:10} | Hits: {num_hits}/{num_total} (40%) | Avg Throughput: {throughput:7.2f} GB/s", flush=True)
        return throughput
    return throughput

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", type=str, help="Comma-separated systems")
    parser.add_argument("--blocks-per-rank", type=int, default=160, help="Blocks per rank (default 160, ~26GB)")
    args = parser.parse_args()
    
    systems_to_test = args.systems.split(',')
    num_blocks = args.blocks_per_rank

    loader = RealKVLoader()
    # Contention Scenario: All ranks read SAME 40 blocks simultaneously
    num_shared = 40
    read_ids = loader.block_ids[:num_shared]
    
    print_rank0(f" Tiered Config: GPU(40% hit) + Storage(60% miss), {num_shared} shared blocks")
    print_rank0(f" [Global] Pre-loading {len(read_ids)} shared blocks to memory...")
    loaded_data_dict = loader.read_blocks(read_ids)
    
    import gc
    gc.collect()
    
    print_rank0(f" Running Tiered Comparison ({world} Nodes)")
    print_rank0("="*80)
    print_rank0(f"{'System':12} | {'Hit Rate':>15} | {'Avg Throughput (GB/s)':>25}")
    print_rank0("-" * 65)
    
    final_results = {}
    
    for name in systems_to_test:
        adapter = None
        try:
            # All baselines now use SHARED paths to trigger Lustre contention
            if name == "Cascade":
                adapter = CascadeAdapter()
            elif name == "HDF5": 
                adapter = HDF5Adapter(shared=True)
            elif name == "LMCache": 
                adapter = LMCacheAdapter(shared=True)
            elif name == "vLLM-GPU":
                adapter = vLLMGPUAdapter(shared=True)
            elif name == "PDC":
                adapter = PDCAdapter(shared=True)
            
            if adapter:
                tp = run_tiered_bench(name, adapter, [], read_ids, loaded_data_dict)
                if rank == 0: final_results[name] = tp
                # Safe cleanup to avoid multicore race condition
                try: adapter.cleanup()
                except: pass
        except Exception as e:
            print(f"Error testing {name} on rank {rank}: {e}")
            import traceback
            traceback.print_exc()

    if rank == 0:
        print("\n" + "="*80)
        print(f" Final Tiered Comparison Summary ({world} Nodes)")
        print("="*80)
        print(f"{'System':12} | {'Avg Throughput (GB/s)':>25}")
        print("-" * 50)
        # Sort results by throughput descending
        sorted_res = sorted(final_results.items(), key=lambda x: x[1], reverse=True)
        for name, tp in sorted_res:
            print(f"{name:12} | {tp:25.2f}")
        print("="*80)

if __name__ == "__main__":
    main()
