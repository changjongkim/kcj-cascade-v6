#!/usr/bin/env python3
import os
import sys
import time
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

def generate_block(seed, size=160*1024):
    np.random.seed(seed)
    return np.random.randint(0, 255, size, dtype=np.uint8)

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
        cfg.gpu_capacity_per_device = 1 * 1024**3 # 1GB/GPU
        cfg.dram_capacity = 2 * 1024**3           # 2GB DRAM/node
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        self.lustre_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cascade_{mode}_r{rank}")
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
        self.path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_{mode}_r{rank}.h5"
        self.file = h5py.File(self.path, 'w')
    def put(self, key, data):
        self.file.create_dataset(key, data=data)
        self.file.flush()
    def get(self, key, out):
        dset = self.file[key]
        dset.read_direct(out)
    def barrier(self): pass # HDF5 is local here
    def cleanup(self):
        self.file.close()
        if os.path.exists(self.path): os.remove(self.path)

class vLLMGPUAdapter(BaseStore):
    def __init__(self, mode):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/vllm_{mode}_r{rank}")
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
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/pdc_{mode}_r{rank}")
        self.dir.mkdir(parents=True, exist_ok=True)
    def put(self, key, data):
        path = self.dir / f"{key}.bin"
        with open(path, 'wb') as f: f.write(data); f.flush()
    def get(self, key, out):
        p = self.dir / f"{key}.bin"
        with open(p, 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
    def barrier(self): pass
    def cleanup(self):
        import shutil
        if self.dir.exists(): shutil.rmtree(self.dir)

class LMCacheAdapter(BaseStore):
    def __init__(self, mode):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/lmcache_{mode}_r{rank}")
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
    parser.add_argument("--systems", default="Cascade,HDF5,PDC,LMCache,vLLM-GPU")
    parser.add_argument("--block-size-kb", type=int, default=160)
    args = parser.parse_args()

    systems = args.systems.split(",")
    block_size = args.block_size_kb * 1024
    
    if args.mode == "strong":
        # Total data: 12.5 GB (80,000 blocks)
        TOTAL_BLOCKS = 80000
        my_blocks = TOTAL_BLOCKS // world
        total_data_gb = (TOTAL_BLOCKS * block_size) / 1024**3
    else:
        # Weak scaling: 10,000 blocks per rank (~1.5 GB per rank)
        my_blocks = 10000
        TOTAL_BLOCKS = my_blocks * world
        total_data_gb = (TOTAL_BLOCKS * block_size) / 1024**3

    print_rank0(f"SCALING BENCHMARK | Mode: {args.mode.upper()} | World: {world} | Total Data: {total_data_gb:.2f} GB")
    print_rank0("="*80)
    print_rank0(f"{'System':12} | {'Write BW (GB/s)':>18} | {'Read BW (GB/s)':>18}")
    print_rank0("-" * 60)

    for name in systems:
        adapter = None
        if name == "Cascade": adapter = CascadeAdapter(args.mode)
        elif name == "HDF5": adapter = HDF5Adapter(args.mode)
        elif name == "PDC": adapter = PDCAdapter(args.mode)
        elif name == "LMCache": adapter = LMCacheAdapter(args.mode)
        elif name == "vLLM-GPU": adapter = vLLMGPUAdapter(args.mode)
        
        if not adapter: continue

        keys = [f"b_{i:06d}" for i in range(rank * my_blocks, (rank + 1) * my_blocks)]
        data_block = generate_block(rank, block_size)

        # Write Phase
        adapter.barrier()
        t0 = time.time()
        for k in keys:
            adapter.put(k, data_block)
        adapter.barrier()
        t_write = time.time() - t0
        write_bw = total_data_gb / t_write

        # Read Phase
        adapter.barrier()
        t1 = time.time()
        for k in keys:
            buf = np.empty(block_size, dtype=np.uint8)
            adapter.get(k, buf)
        adapter.barrier()
        t_read = time.time() - t1
        read_bw = total_data_gb / t_read

        print_rank0(f"{name:12} | {write_bw:18.2f} | {read_bw:18.2f}")
        adapter.cleanup()

    print_rank0("="*80)

if __name__ == "__main__":
    main()
