#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import argparse
from pathlib import Path

# MPI Configuration
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = comm.Get_size()

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
    print("Error: cascade_cpp module not found.")
    sys.exit(1)

def print_rank0(msg):
    if rank == 0:
        print(msg, flush=True)

# ======================================== ==============
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

# Global Store for Barrier
GLOBAL_STORE = None

def init_global_store(mode):
    global GLOBAL_STORE
    if GLOBAL_STORE is None:
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 128 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_cont_{mode}"
        GLOBAL_STORE = cascade_cpp.DistributedStore(cfg)
    return GLOBAL_STORE

# MPI Shared Barrier replacement
def mpi_barrier():
    if GLOBAL_STORE:
        GLOBAL_STORE.barrier()

class CascadeAdapter(BaseStore):
    def __init__(self, mode):
        self.mode = mode
        # Re-use global store
        self.store = GLOBAL_STORE
        self.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_cont_{mode}"

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
        if rank == 0:
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

def get_model_config(model_name):
    # (Layers, KV Heads, Head Dim, Tokens per block)
    configs = {
        "llama-3-70b": (80, 8, 128, 1024),
        "qwen-2.5-72b": (80, 8, 128, 1024),
        "qwen-2.5-32b": (64, 8, 128, 1024),
        "qwen-2.5-7b": (28, 4, 128, 1024)
    }
    L, H, D, T = configs.get(model_name, configs["llama-3-70b"])
    # 2 (fp16) * 2 (K and V) * L * H * D * T
    return 2 * 2 * L * H * D * T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["strong", "weak"], required=True)
    parser.add_argument("--data-type", choices=["synthetic", "real"], required=True)
    parser.add_argument("--model", default="llama-3-70b", choices=["llama-3-70b", "qwen-2.5-72b", "qwen-2.5-32b", "qwen-2.5-7b"])
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache")
    args = parser.parse_args()
    
    # Initialize Global Store for Barrier (MUST be done first)
    barrier_store = init_global_store(args.mode)
    barrier_store.barrier() # Sync start

    systems = args.systems.split(",")
    block_size = get_model_config(args.model)
    read_keys = []
    
    if args.data_type == "real":
        loader = RealKVLoader()
        # For Qwen, if real data is not available, we use synthetic generation with realistic size
        is_qwen = "qwen" in args.model
        
        if args.mode == "strong":
            TOTAL_BLOCKS = 128
            read_keys = loader.block_ids[:TOTAL_BLOCKS] if not is_qwen else [f"qwen_b{i}" for i in range(TOTAL_BLOCKS)]
            write_keys = read_keys if rank == 0 else []
        else:
            GLOBAL_BLOCKS = 40
            LOCAL_BLOCKS = 20
            global_keys = (loader.block_ids[:GLOBAL_BLOCKS] if not is_qwen else [f"qwen_g{i}" for i in range(GLOBAL_BLOCKS)])
            local_keys = [f"rank{rank}_b{i}" for i in range(LOCAL_BLOCKS)]
            read_keys = global_keys + local_keys
            write_keys = global_keys if rank == 0 else local_keys
        
        data_cache = {}
        for k in read_keys:
            if not is_qwen and k in loader.block_ids:
                data_cache[k] = loader.read_block(k)
            else:
                data_cache[k] = generate_block(abs(hash(k)) % (2**32), block_size)
    else:
        # Synthetic
        if args.mode == "strong":
            TOTAL_BLOCKS = 1000 # 1000 blocks
            read_keys = [f"shared_b{i}" for i in range(TOTAL_BLOCKS)]
            write_keys = read_keys if rank == 0 else []
        else:
            GLOBAL_BLOCKS = 200
            LOCAL_BLOCKS = 100
            global_keys = [f"global_b{i}" for i in range(GLOBAL_BLOCKS)]
            local_keys = [f"node{rank}_b{i}" for i in range(LOCAL_BLOCKS)]
            read_keys = global_keys + local_keys
            write_keys = global_keys if rank == 0 else local_keys
        
        data_cache = {k: generate_block(abs(hash(k)) % (2**32), block_size) for k in read_keys}

    total_read_data_gb = (len(read_keys) * world * block_size) / 1024**3
    print_rank0(f"=== {args.model.upper()} Benchmark Configuration ===")
    print_rank0(f"Block Size: {block_size / 1024**2:.2f} MB")
    
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
        local_avg_lat = np.mean(latencies)
        
        # Gather per-node stats
        all_bw = comm.gather(aggr_bw / world, root=0) # Estimated per-rank share
        all_lats = comm.gather(local_avg_lat, root=0)

        if rank == 0:
            print_rank0(f"{name:12} | {aggr_bw:22.2f} | {local_avg_lat:22.2f}")
            node_bw_str = ", ".join([f"{b:.1f}" for b in all_bw])
            node_lat_str = ", ".join([f"{l:.1f}" for l in all_lats])
            print_rank0(f"  [Detail] Per-node BW:  {node_bw_str}")
            print_rank0(f"  [Detail] Per-node Lat: {node_lat_str}")
        adapter.cleanup()

    print_rank0("="*90)

if __name__ == "__main__":
    main()
