#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import argparse
from pathlib import Path

# MPI Configuration
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))

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
# Real Data Loader
# ============================================================

class RealKVLoader:
    def __init__(self, base_dir="/pscratch/sd/s/sgkim/cascade_kv_cache"):
        self.base_dir = Path(base_dir)
        index_path = self.base_dir / "global_index.json"
        with open(index_path, 'r') as f:
            data = json.load(f)
            self.all_blocks = data['blocks']
            self.block_ids = list(self.all_blocks.keys())
            
    def get_blocks(self, start_idx, count):
        return self.block_ids[start_idx : start_idx + count]

    def read_block(self, block_id):
        loc = self.all_blocks[block_id]
        path = self.base_dir / loc['file']
        with open(path, 'rb') as f:
            f.seek(loc['offset'])
            return np.frombuffer(f.read(loc['size']), dtype=np.uint8)

# ============================================================
# Storage Adapters
# ============================================================

class BaseStore:
    def put(self, key, data): pass
    def get(self, key, out): pass
    def barrier(self): pass
    def cleanup(self): pass

class CascadeAdapter(BaseStore):
    def __init__(self, mode):
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 128 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        self.lustre_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/real_cascade_{mode}_r{rank}")
        self.lustre_dir.mkdir(parents=True, exist_ok=True)
        cfg.lustre_path = str(self.lustre_dir)
        self.store = cascade_cpp.DistributedStore(cfg)
    def put(self, key, data): return self.store.put(key, data)
    def get(self, key, out): return self.store.get(key, out)
    def barrier(self): self.store.barrier()
    def cleanup(self):
        import shutil
        if self.lustre_dir.exists(): shutil.rmtree(self.lustre_dir)

class HDF5Adapter(BaseStore):
    def __init__(self, mode):
        import h5py
        self.path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/real_hdf5_{mode}_r{rank}.h5"
        self.file = h5py.File(self.path, 'w')
    def put(self, key, data):
        if key not in self.file: self.file.create_dataset(key, data=data)
    def get(self, key, out):
        dset = self.file[key]
        dset.read_direct(out)
    def barrier(self): pass
    def cleanup(self):
        self.file.close()
        if os.path.exists(self.path): os.remove(self.path)

class vLLMGPUAdapter(BaseStore):
    def __init__(self, mode):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/real_vllm_{mode}_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    def put(self, key, data):
        with open(self.dir / f"{key}.pt", 'wb') as f: f.write(data)
    def get(self, key, out):
        with open(self.dir / f"{key}.pt", 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
    def barrier(self): pass
    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

class PDCAdapter(BaseStore):
    def __init__(self, mode):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/real_pdc_{mode}_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    def put(self, key, data):
        with open(self.dir / f"{key}.bin", 'wb') as f: f.write(data)
    def get(self, key, out):
        with open(self.dir / f"{key}.bin", 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
    def barrier(self): pass
    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

class LMCacheAdapter(BaseStore):
    def __init__(self, mode):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/real_lmcache_{mode}_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    def put(self, key, data): data.tofile(self.dir / f"{key}.bin")
    def get(self, key, out): out[:] = np.fromfile(self.dir / f"{key}.bin", dtype=np.uint8)
    def barrier(self): pass
    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

# ============================================================
# Main Logic
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["strong", "weak"], required=True)
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache")
    args = parser.parse_args()

    loader = RealKVLoader()
    systems = args.systems.split(",")
    
    if args.mode == "strong":
        # Strong scaling: Fixed total 256 blocks (~40GB total real data)
        TOTAL_BLOCKS = 256
        my_count = TOTAL_BLOCKS // world
        my_start = rank * my_count
    else:
        # Weak scaling: 40 blocks per rank (~6.5GB per rank)
        my_count = 40
        my_start = rank * my_count
        TOTAL_BLOCKS = my_count * world

    my_block_ids = loader.get_blocks(my_start, my_count)
    
    # Pre-calculate total data scale
    sample_block = loader.read_block(my_block_ids[0])
    block_size = sample_block.nbytes
    total_data_gb = (TOTAL_BLOCKS * block_size) / 1024**3

    print_rank0(f"REAL WORKLOAD SCALING | Mode: {args.mode.upper()} | World: {world} | Total: {total_data_gb:.2f} GB")
    print_rank0("="*85)
    print_rank0(f"{'System':12} | {'Write BW (GB/s)':>18} | {'Read BW (GB/s)':>18} | {'Avg Latency (ms)':>18}")
    print_rank0("-" * 80)

    # Cache loaded data to avoid Lustre loading noise in the middle of benchmark
    data_cache = {bid: loader.read_block(bid) for bid in my_block_ids}

    for name in systems:
        adapter = None
        if name == "Cascade": adapter = CascadeAdapter(args.mode)
        elif name == "HDF5": adapter = HDF5Adapter(args.mode)
        elif name == "vLLM-GPU": adapter = vLLMGPUAdapter(args.mode)
        elif name == "PDC": adapter = PDCAdapter(args.mode)
        elif name == "LMCache": adapter = LMCacheAdapter(args.mode)
        
        if not adapter: continue

        # Write Phase
        adapter.barrier()
        t0 = time.time()
        for idx, bid in enumerate(my_block_ids):
            adapter.put(bid, data_cache[bid])
        adapter.barrier()
        t_write = time.time() - t0
        write_bw = total_data_gb / t_write

        # Read Phase
        latencies = []
        adapter.barrier()
        t1 = time.time()
        for bid in my_block_ids:
            tl0 = time.time()
            buf = np.empty(block_size, dtype=np.uint8)
            adapter.get(bid, buf)
            latencies.append((time.time() - tl0) * 1000)
        adapter.barrier()
        t_read = time.time() - t1
        read_bw = total_data_gb / t_read
        avg_lat = np.mean(latencies)

        print_rank0(f"{name:12} | {write_bw:18.2f} | {read_bw:18.2f} | {avg_lat:18.2f}")
        adapter.cleanup()

    print_rank0("="*85)

if __name__ == "__main__":
    main()
