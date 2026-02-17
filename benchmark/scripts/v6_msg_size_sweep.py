import os
import sys
import time
import numpy as np
import argparse
from pathlib import Path

# Path setup
default_build = 'cascade_Code/cpp/build_cascade_cpp'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.getcwd(), build_dir)
sys.path.insert(0, build_dir)

try:
    import cascade_cpp
except ImportError:
    cascade_cpp = None
    print("Error: cascade_cpp module not found. Please ensure CASCADE_BUILD_DIR is correct.")
    sys.exit(1)

# MPI Configuration via SLURM
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))

# Global barrier helper
GLOBAL_BARRIER_STORE = None

def mpi_barrier():
    if GLOBAL_BARRIER_STORE:
        GLOBAL_BARRIER_STORE.barrier()

def print_rank0(msg):
    if rank == 0:
        print(msg, flush=True)

class BaseStore:
    def put(self, key, data): pass
    def get(self, key, out): pass
    def barrier(self): pass
    def cleanup(self): pass

class CascadeAdapter(BaseStore):
    def __init__(self, msg_size_name):
        global GLOBAL_BARRIER_STORE
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 128 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_msg_{msg_size_name}"
        self.store = cascade_cpp.DistributedStore(cfg)
        self.lustre_path = cfg.lustre_path 
        # Register as global barrier store
        GLOBAL_BARRIER_STORE = self.store
        
    def put(self, key, data): self.store.put(str(key), data)
    def get(self, key, out): return self.store.get(str(key), out)[0]
    def barrier(self): self.store.barrier()
    def cleanup(self):
        self.barrier()
        # Only rank 0 cleans up the lustre path if it exists
        if rank == 0:
            import shutil
            if os.path.exists(self.lustre_path): shutil.rmtree(self.lustre_path)

class LMCacheAdapter(BaseStore):
    def __init__(self, msg_size_name):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lmcache_msg_{msg_size_name}")
        if rank == 0: self.dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()
    def put(self, key, data):
        if rank == 0:
            with open(self.dir / str(key), 'wb') as f: f.write(data)
    def get(self, key, out):
        p = self.dir / str(key)
        if p.exists():
            with open(p, 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
            return True
        return False
    def barrier(self): mpi_barrier()
    def cleanup(self):
        mpi_barrier()
        import shutil
        if rank == 0 and self.dir.exists(): shutil.rmtree(self.dir)

def run_bench(adapter, msg_size, label, num_blocks=10):
    # Prepare data (Rank 0 generates, others just alloc buffer)
    # We use 'label' in key to avoid cross-test conflicts/caching
    keys = [f"{label}_{i}" for i in range(num_blocks)]
    
    # Write phase (Rank 0 writes)
    if rank == 0:
        # Use a different seed per call to ensure no pattern reuse
        np.random.seed(int(time.time() * 1000) % 2**32)
        data = np.random.randint(0, 255, msg_size, dtype=np.uint8)
        for k in keys:
            adapter.put(k, data)
    adapter.barrier()
    
    # Read phase (All ranks read same data - Broadcast Pattern)
    t_start = time.time()
    for k in keys:
        buf = np.empty(msg_size, dtype=np.uint8)
        ret = adapter.get(k, buf)
    adapter.barrier()
    t_end = time.time()
    
    duration = t_end - t_start
    total_data_gb = (msg_size * num_blocks * world) / (1024**3)
    bw = total_data_gb / duration if duration > 0 else 0
    return bw

def main():
    msg_sizes = {
        "64KB": 64 * 1024,
        "256KB": 256 * 1024,
        "1MB": 1024 * 1024,
        "10MB": 10 * 1024 * 1024,
        "16MB": 16 * 1024 * 1024,
        "100MB": 100 * 1024 * 1024,
        "160MB": 160 * 1024 * 1024
    }
    
    print_rank0(f"Message Size Sweep | Nodes: {world} | Tasks: {world}")
    print_rank0("="*70)
    print_rank0(f"{'Size':10} | {'Cascade (GB/s)':>20} | {'LMCache (GB/s)':>20}")
    print_rank0("-" * 70)
    
    # Initialize Cascade once
    cas = CascadeAdapter("all_sizes")
    
    for label, size in msg_sizes.items():
        # Cascade
        cas_bw = run_bench(cas, size, f"cas_{label}")
        
        # LMCache
        lm = LMCacheAdapter(label)
        lm_bw = run_bench(lm, size, f"lm_{label}")
        lm.cleanup()
        
        print_rank0(f"{label:10} | {cas_bw:20.2f} | {lm_bw:20.2f}")
    
    cas.cleanup()

if __name__ == "__main__":
    main()
