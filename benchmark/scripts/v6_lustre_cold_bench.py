#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import argparse
from pathlib import Path

# MPI Setup
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

def clear_page_cache(file_path):
    """Evict file from OS page cache using posix_fadvise."""
    try:
        fd = os.open(file_path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception as e:
        if rank == 0: print(f"Warning: Failed to clear cache for {file_path}: {e}")

# ============================================================
# Adapters
# ============================================================

class BaseLustreStore:
    def write(self, key, data): pass
    def read(self, key, out): pass
    def get_file_path(self, key): return None
    def cleanup(self): pass

class CascadeLustreAdapter(BaseLustreStore):
    def __init__(self):
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 0 # Force Tier 5 focus
        cfg.dram_capacity = 0           # Force Tier 5 focus
        self.lustre_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_cold_cascade_r{rank}")
        self.lustre_dir.mkdir(parents=True, exist_ok=True)
        cfg.lustre_path = str(self.lustre_dir)
        self.store = cascade_cpp.DistributedStore(cfg)
    
    def write(self, key, data):
        self.store.put(key, data)
    
    def read(self, key, out):
        self.store.get(key, out)
    
    def get_file_paths(self, keys):
        # In Cascade, files are inside the lustre_dir
        return list(self.lustre_dir.glob("*"))

    def cleanup(self):
        import shutil
        if self.lustre_dir.exists(): shutil.rmtree(self.lustre_dir)

class HDF5LustreAdapter(BaseLustreStore):
    def __init__(self):
        import h5py
        self.path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_cold_r{rank}.h5"
        self.file = h5py.File(self.path, 'w')
    
    def write(self, key, data):
        self.file.create_dataset(key, data=data)
        self.file.flush()
    
    def read(self, key, out):
        dset = self.file[key]
        dset.read_direct(out)
    
    def get_file_paths(self, keys):
        return [self.path]

    def cleanup(self):
        self.file.close()
        if os.path.exists(self.path): os.remove(self.path)

class vLLMGPUPlusAdapter(BaseLustreStore):
    """Standard Posix File I/O"""
    def __init__(self):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_cold_vllm_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def write(self, key, data):
        with open(self.dir / f"{key}.bin", 'wb') as f:
            f.write(data)
    
    def read(self, key, out):
        with open(self.dir / f"{key}.bin", 'rb') as f:
            out[:] = np.frombuffer(f.read(), dtype=np.uint8)
    
    def get_file_paths(self, keys):
        return [str(self.dir / f"{k}.bin") for k in keys]

    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

class PDCLustreAdapter(BaseLustreStore):
    """Simulates PDC (Optimized Persistence)"""
    def __init__(self):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_cold_pdc_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def write(self, key, data):
        # PDC simulation uses O_SYNC-like behavior
        path = self.dir / f"{key}.bin"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.write(fd, data.tobytes())
        os.fsync(fd)
        os.close(fd)
    
    def read(self, key, out):
        with open(self.dir / f"{key}.bin", 'rb') as f:
            out[:] = np.frombuffer(f.read(), dtype=np.uint8)

    def get_file_paths(self, keys):
        return [str(self.dir / f"{k}.bin") for k in keys]

    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

class LMCacheLustreAdapter(BaseLustreStore):
    def __init__(self):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lustre_cold_lmcache_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def write(self, key, data):
        # Uses numpy.tofile
        data.tofile(self.dir / f"{key}.bin")
    
    def read(self, key, out):
        out[:] = np.fromfile(self.dir / f"{key}.bin", dtype=np.uint8)

    def get_file_paths(self, keys):
        return [str(self.dir / f"{k}.bin") for k in keys]

    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

# ============================================================
# Main Execution
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size-mb", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--systems", type=str, default="Cascade,HDF5,vLLM-GPU,PDC,LMCache")
    args = parser.parse_args()

    block_size = args.block_size_mb * 1024 * 1024
    num_blocks = args.num_blocks
    systems_to_test = args.systems.split(",")

    data_to_write = [np.random.randint(0, 256, block_size, dtype=np.uint8) for _ in range(num_blocks)]
    block_keys = [f"block_{i}" for i in range(num_blocks)]
    total_gb = (block_size * num_blocks) / 1024**3

    print_rank0(f"Lustre Cold Storage Benchmark: {world} nodes, {num_blocks} blocks of {args.block_size_mb}MB ({total_gb:.2f} GB/rank)")
    print_rank0("="*80)
    print_rank0(f"{'System':12} | {'Write (GB/s)':>12} | {'Hot Read':>12} | {'Cold Read':>12}")
    print_rank0("-" * 65)

    for name in systems_to_test:
        adapter = None
        if name == "Cascade": adapter = CascadeLustreAdapter()
        elif name == "HDF5": adapter = HDF5LustreAdapter()
        elif name == "vLLM-GPU": adapter = vLLMGPUPlusAdapter()
        elif name == "PDC": adapter = PDCLustreAdapter()
        elif name == "LMCache": adapter = LMCacheLustreAdapter()
        
        if not adapter: continue

        # 1. Write
        t0 = time.time()
        for i in range(num_blocks):
            adapter.write(block_keys[i], data_to_write[i])
        t_write = time.time() - t0
        write_bw = total_gb / t_write

        # 2. Hot Read (Cached)
        t1 = time.time()
        for i in range(num_blocks):
            out = np.empty(block_size, dtype=np.uint8)
            adapter.read(block_keys[i], out)
        t_hot = time.time() - t1
        hot_bw = total_gb / t_hot

        # 3. Cold Read (Evicted)
        paths = adapter.get_file_paths(block_keys)
        for p in paths: clear_page_cache(str(p))
        
        t2 = time.time()
        for i in range(num_blocks):
            out = np.empty(block_size, dtype=np.uint8)
            adapter.read(block_keys[i], out)
        t_cold = time.time() - t2
        cold_bw = total_gb / t_cold

        if rank == 0:
            print(f"{name:12} | {write_bw:12.2f} | {hot_bw:12.2f} | {cold_bw:12.2f}", flush=True)
        
        adapter.cleanup()

if __name__ == "__main__":
    main()
