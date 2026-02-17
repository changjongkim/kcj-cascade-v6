#!/usr/bin/env python3
import os
import sys
import time
import random
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

# ============================================================
# Generic SHM Cache Wrapper
# ============================================================

class SHMCacheLayer:
    def __init__(self, capacity_blocks, block_size, backend_adapter):
        self.capacity = capacity_blocks
        self.block_size = block_size
        self.backend = backend_adapter
        self.shm_dir = Path(f"/dev/shm/cascade_bench_r{rank}")
        self.shm_dir.mkdir(parents=True, exist_ok=True)
        self.cache_keys = [] # Simple FIFO for simulation

    def get(self, key, out):
        shm_path = self.shm_dir / f"{key}.shm"
        
        # 1. Check SHM (Hit)
        if shm_path.exists():
            with open(shm_path, 'rb') as f:
                out[:] = np.frombuffer(f.read(), dtype=np.uint8)
            return True, "HIT"
        
        # 2. SHM MISS -> Fetch from Backend
        success = self.backend.read(key, out)
        if success:
            # 3. Populate SHM (Evict if full)
            if len(self.cache_keys) >= self.capacity:
                old_key = self.cache_keys.pop(0)
                old_path = self.shm_dir / f"{old_key}.shm"
                if old_path.exists(): os.remove(old_path)
            
            with open(shm_path, 'wb') as f:
                f.write(out.tobytes())
            self.cache_keys.append(key)
            return True, "MISS"
        
        return False, "ERROR"

    def cleanup(self):
        import shutil
        if self.shm_dir.exists(): shutil.rmtree(self.shm_dir)
        self.backend.cleanup()

# ============================================================
# Backend Adapters (Reused from Cold Bench)
# ============================================================

class CascadeBackend:
    def __init__(self):
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 0 
        cfg.dram_capacity = 0           
        self.lustre_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/shm_back_cascade_r{rank}")
        self.lustre_dir.mkdir(parents=True, exist_ok=True)
        cfg.lustre_path = str(self.lustre_dir)
        self.store = cascade_cpp.DistributedStore(cfg)
    def write(self, key, data): self.store.put(key, data)
    def read(self, key, out): self.store.get(key, out); return True
    def cleanup(self):
        import shutil
        if self.lustre_dir.exists(): shutil.rmtree(self.lustre_dir)

class HDF5Backend:
    def __init__(self):
        import h5py
        self.path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/shm_back_r{rank}.h5"
        self.file = h5py.File(self.path, 'w')
    def write(self, key, data): self.file.create_dataset(key, data=data); self.file.flush()
    def read(self, key, out): dset = self.file[key]; dset.read_direct(out); return True
    def cleanup(self): self.file.close(); os.remove(self.path)

class PosixBackend:
    def __init__(self):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/shm_back_posix_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    def write(self, key, data):
        with open(self.dir / f"{key}.bin", 'wb') as f: f.write(data)
    def read(self, key, out):
        with open(self.dir / f"{key}.bin", 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
        return True
    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

class PDCBackend:
    def __init__(self):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/shm_back_pdc_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    def write(self, key, data):
        path = self.dir / f"{key}.bin"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.write(fd, data.tobytes())
        os.fsync(fd); os.close(fd)
    def read(self, key, out):
        with open(self.dir / f"{key}.bin", 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
        return True
    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

class LMCacheBackend:
    def __init__(self):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/shm_back_lmcache_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    def write(self, key, data): data.tofile(self.dir / f"{key}.bin")
    def read(self, key, out): out[:] = np.fromfile(self.dir / f"{key}.bin", dtype=np.uint8); return True
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
    parser.add_argument("--cache-size", type=int, default=5)
    parser.add_argument("--num-access", type=int, default=50)
    parser.add_argument("--systems", type=str, default="Cascade,HDF5,vLLM-GPU")
    args = parser.parse_args()

    block_size = args.block_size_mb * 1024 * 1024
    systems_to_test = args.systems.split(",")
    block_keys = [f"block_{i}" for i in range(args.num_blocks)]
    
    # Generate deterministic 60% hit rate (30 hits out of 50 accesses)
    random.seed(42 + rank)
    
    # Simple strategy: 30 accesses to blocks 0-4 (SHM capacity), 20 to 5-9 (Misses)
    hits_needed = int(args.num_access * 0.6) # 30
    misses_needed = args.num_access - hits_needed # 20
    
    pattern = ([random.choice(block_keys[:args.cache_size]) for _ in range(hits_needed)] + 
               [random.choice(block_keys[args.cache_size:]) for _ in range(misses_needed)])
    random.shuffle(pattern)
    access_pattern = pattern

    print_rank0(f"SHM Cache Layer Benchmark: {world} nodes, {args.num_access} accesses")
    print_rank0(f"Config: {args.block_size_mb}MB blocks, SHM Capacity: {args.cache_size} blocks (Target Hit Rate: 60%)")
    print_rank0("="*80)
    print_rank0(f"{'System':12} | {'Avg Latency (ms)':>18} | {'Hit Rate (%)':>15} | {'Throughput (GB/s)':>18}")
    print_rank0("-" * 75)

    data_sample = np.random.randint(0, 256, block_size, dtype=np.uint8)

    for name in systems_to_test:
        backend = None
        if name == "Cascade": backend = CascadeBackend()
        elif name == "HDF5": backend = HDF5Backend()
        elif name == "vLLM-GPU": backend = PosixBackend()
        elif name == "PDC": backend = PDCBackend()
        elif name == "LMCache": backend = LMCacheBackend()
        
        if not backend: continue

        # Initial Write to Backend
        for k in block_keys: backend.write(k, data_sample)

        shm_cache = SHMCacheLayer(args.cache_size, block_size, backend)
        
        hits = 0
        latencies = []
        out = np.empty(block_size, dtype=np.uint8)

        # Execution Loop
        t_start = time.time()
        for key in access_pattern:
            t0 = time.time()
            success, status = shm_cache.get(key, out)
            latencies.append((time.time() - t0) * 1000)
            if status == "HIT": hits += 1
        
        total_time = time.time() - t_start
        avg_lat = np.mean(latencies)
        hit_rate = (hits / args.num_access) * 100
        throughput = (block_size * args.num_access) / (total_time * 1024**3)

        if rank == 0:
            print(f"{name:12} | {avg_lat:18.2f} | {hit_rate:15.1f} | {throughput:18.2f}", flush=True)

        shm_cache.cleanup()

if __name__ == "__main__":
    main()
