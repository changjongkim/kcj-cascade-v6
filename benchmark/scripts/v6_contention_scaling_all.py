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

# Path setup
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
# Helpers
# ============================================================

class RealKVLoader:
    def __init__(self, base_dir="/pscratch/sd/s/sgkim/cascade_kv_cache"):
        self.base_dir = Path(base_dir)
        index_path = self.base_dir / "global_index.json"
        with open(index_path, 'r') as f:
            data = json.load(f)
            self.all_blocks = data['blocks']
            self.block_ids = list(self.all_blocks.keys())
            
    def read_block(self, block_id):
        loc = self.all_blocks[block_id]
        path = self.base_dir / loc['file']
        with open(path, 'rb') as f:
            f.seek(loc['offset'])
            return np.frombuffer(f.read(loc['size']), dtype=np.uint8)

def generate_block(seed, size):
    np.random.seed(seed)
    return np.random.randint(0, 255, size, dtype=np.uint8)

# ============================================================
# Storage Adapters (Common)
# ============================================================

class BaseStore:
    def put(self, key, data): pass
    def get(self, key, out): pass
    def barrier(self): pass
    def cleanup(self): pass

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
if not MPI.Is_initialized():
    # Cascade V6 backend requires THREAD_MULTIPLE for RMA/Async threads
    MPI.Init_thread(MPI.THREAD_MULTIPLE)

# MPI Shared Barrier
def mpi_barrier():
    MPI.COMM_WORLD.Barrier()

class CascadeAdapter(BaseStore):
    def __init__(self, mode):
        self.mode = mode
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 128 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        self.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_cont_{mode}"
        cfg.lustre_path = self.lustre_path
        self.store = cascade_cpp.DistributedStore(cfg)

    def put(self, key, data):
        self.store.put(str(key), data)

    def get(self, key, out):
        found, size = self.store.get(str(key), out)
        return found

    def barrier(self):
        self.store.barrier()

    def cleanup(self):
        self.barrier()
        # Only rank 0 cleans up the lustre path if it exists
        if self.store.rank == 0:
            import shutil
            if os.path.exists(self.lustre_path):
                try:
                    shutil.rmtree(self.lustre_path)
                except:
                    pass
        
class HDF5Adapter(BaseStore):
    def __init__(self, mode):
        import h5py
        self.h5py = h5py
        # We separate shared (rank 0 writes) and local (each rank writes)
        self.shared_path = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_shared_cont_{mode}.h5")
        self.local_path = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_local_cont_{mode}_r{rank}.h5")
        self.file_write = None
        self.file_read_shared = None
        self.file_read_local = None

    def put(self, key, data):
        # If it's a shared key (doesn't contain 'rank' or 'node'), only rank 0 writes to shared file
        is_shared = "rank" not in str(key) and "node" not in str(key)
        if is_shared:
            if rank == 0:
                if not self.file_write: self.file_write = self.h5py.File(self.shared_path, 'a')
                if str(key) not in self.file_write:
                    self.file_write.create_dataset(str(key), data=data)
                    self.file_write.flush()
        else:
            # Local keys: each rank writes to its own file
            if not self.file_write: self.file_write = self.h5py.File(self.local_path, 'a')
            if str(key) not in self.file_write:
                self.file_write.create_dataset(str(key), data=data)
                self.file_write.flush()

    def open_for_read(self):
        if self.file_write: self.file_write.close()
        self.file_write = None
        if self.shared_path.exists():
            self.file_read_shared = self.h5py.File(self.shared_path, 'r')
        if self.local_path.exists():
            self.file_read_local = self.h5py.File(self.local_path, 'r')

    def get(self, key, out):
        # Try local file first, then shared file
        if self.file_read_local and str(key) in self.file_read_local:
            out[:] = self.file_read_local[str(key)][:]
            return True
        if self.file_read_shared and str(key) in self.file_read_shared:
            out[:] = self.file_read_shared[str(key)][:]
            return True
        return False

    def barrier(self): mpi_barrier()
    def cleanup(self):
        mpi_barrier()
        if self.file_read_shared: self.file_read_shared.close()
        if self.file_read_local: self.file_read_local.close()
        if self.file_write: self.file_write.close()
        if rank == 0 and self.shared_path.exists(): self.shared_path.unlink()
        if self.local_path.exists(): self.local_path.unlink()

class PosixAdapter(BaseStore):
    def __init__(self, name, mode):
        self.shared_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_shared_cont_{mode}")
        self.local_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_local_cont_{mode}_r{rank}")
        if rank == 0: self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()

    def put(self, key, data):
        is_shared = "rank" not in str(key) and "node" not in str(key)
        if is_shared:
            if rank == 0:
                with open(self.shared_dir / str(key), 'wb') as f: f.write(data)
        else:
            with open(self.local_dir / str(key), 'wb') as f: f.write(data)

    def get(self, key, out):
        p_local = self.local_dir / str(key)
        p_shared = self.shared_dir / str(key)
        if p_local.exists():
            with open(p_local, 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
            return True
        if p_shared.exists():
            with open(p_shared, 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
            return True
        return False

    def barrier(self): mpi_barrier()
    def cleanup(self):
        import shutil
        mpi_barrier()
        if rank == 0 and self.shared_dir.exists(): shutil.rmtree(self.shared_dir)
        if self.local_dir.exists(): shutil.rmtree(self.local_dir)

# ============================================================
# Main Logic
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["strong", "weak"], required=True)
    parser.add_argument("--data-type", choices=["synthetic", "real"], required=True)
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache")
    args = parser.parse_args()

    systems = args.systems.split(",")
    block_size = 0
    read_keys = []
    
    if args.data_type == "real":
        loader = RealKVLoader()
        # Shared Prefix Contention (Strong): All ranks read SAME 128 blocks (~21GB)
        # Shared Prefix Contention (Weak): Global 40 blocks + Local 20 blocks
        if args.mode == "strong":
            TOTAL_BLOCKS = 128
            read_keys = loader.block_ids[:TOTAL_BLOCKS]
            write_keys = read_keys if rank == 0 else [] # Rank 0 writes for everyone
        else:
            GLOBAL_BLOCKS = 40
            LOCAL_BLOCKS = 20
            global_keys = loader.block_ids[:GLOBAL_BLOCKS]
            local_keys = [f"rank{rank}_b{i}" for i in range(LOCAL_BLOCKS)]
            read_keys = global_keys + local_keys
            write_keys = global_keys if rank == 0 else local_keys
        
        sample = loader.read_block(loader.block_ids[0])
        block_size = sample.nbytes
        data_cache = {bid: loader.read_block(bid) if bid in loader.block_ids else generate_block(abs(hash(bid)) % (2**32), block_size) 
                      for bid in read_keys}
    else:
        # Synthetic
        block_size = 1024 * 1024 # 1MB blocks for faster synthetic
        if args.mode == "strong":
            TOTAL_BLOCKS = 10000 # 10GB total
            read_keys = [f"shared_b{i}" for i in range(TOTAL_BLOCKS)]
            write_keys = read_keys if rank == 0 else []
        else:
            GLOBAL_BLOCKS = 2000
            LOCAL_BLOCKS = 1000
            global_keys = [f"global_b{i}" for i in range(GLOBAL_BLOCKS)]
            local_keys = [f"node{rank}_b{i}" for i in range(LOCAL_BLOCKS)]
            read_keys = global_keys + local_keys
            write_keys = global_keys if rank == 0 else local_keys
        
        data_cache = {k: generate_block(abs(hash(k)) % (2**32), block_size) for k in read_keys}

    total_read_data_gb = (len(read_keys) * world * block_size) / 1024**3
    
    print_rank0(f"CONTENTION SCALING | Type: {args.data_type.upper()} | Mode: {args.mode.upper()} | World: {world}")
    print_rank0(f"Total Aggr. Read Data: {total_read_data_gb:.2f} GB")
    print_rank0("="*90)
    print_rank0(f"{'System':12} | {'Read BW (Aggr. GB/s)':>22} | {'Avg Latency (ms)':>22}")
    print_rank0("-" * 80)

    for name in systems:
        adapter = None
        if name == "Cascade": adapter = CascadeAdapter(args.mode)
        elif name == "HDF5": adapter = HDF5Adapter(args.mode)
        elif name == "vLLM-GPU": adapter = PosixAdapter("vllm", args.mode)
        elif name == "PDC": adapter = PosixAdapter("pdc", args.mode)
        elif name == "LMCache": adapter = PosixAdapter("lmcache", args.mode)
        
        if not adapter: continue

        # 1. Write Phase (Prepare shared data)
        for k in write_keys:
            adapter.put(k, data_cache[k])
        
        # Sync until everyone has written
        adapter.barrier()
        if hasattr(adapter, 'open_for_read'):
            adapter.open_for_read()
        
        # 2. Read Phase (The Contention Test)
        latencies = []
        adapter.barrier()
        t_start = time.time()
        for k in read_keys:
            t0 = time.time()
            buf = np.empty(block_size, dtype=np.uint8)
            adapter.get(k, buf)
            latencies.append((time.time() - t0) * 1000)
        
        if hasattr(adapter, 'barrier'): adapter.barrier()
        t_total = time.time() - t_start
        
        # Aggregate Throughput
        aggr_bw = total_read_data_gb / t_total
        avg_lat = np.mean(latencies)

        print_rank0(f"{name:12} | {aggr_bw:22.2f} | {avg_lat:22.2f}")
        adapter.cleanup()

    print_rank0("="*90)

if __name__ == "__main__":
    main()
