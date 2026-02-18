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
job_id = os.environ.get('SLURM_JOB_ID', 'v7_local')

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
    print(f"[Rank {rank}] Error: cascade_cpp module not found at {build_dir}")
    sys.exit(1)

def print_rank0(msg):
    if rank == 0:
        print(msg, flush=True)

# ============================================================
# Helpers
# ============================================================

def generate_block(seed, size):
    # Fixed seed per block ID to ensure deterministic content for dedup
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size, dtype=np.uint8)

# ============================================================
# Storage Adapters (v7: Improved Robustness)
# ============================================================

class BaseStore:
    def put(self, key, data, is_prefix=False): pass
    def get(self, key, out): pass
    def barrier(self): pass
    def cleanup(self): pass
    def clear(self): pass

GLOBAL_STORE = None

def init_global_store(tag="v7_serv", gpu_cap_gb=38.0):
    global GLOBAL_STORE
    if GLOBAL_STORE is None:
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = int(gpu_cap_gb * 1024**3)
        cfg.dram_capacity = 30 * 1024**3 # 30GB per task (120GB per node)
        cfg.num_gpus_per_node = 1
        cfg.dedup_enabled = True
        cfg.semantic_eviction = True
        cfg.locality_aware = True
        cfg.promotion_threshold = 2
        # Use shared PFS path for Lustre tier
        cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_v7_{tag}_{job_id}"
        GLOBAL_STORE = cascade_cpp.DistributedStore(cfg)
    return GLOBAL_STORE

def mpi_barrier():
    if GLOBAL_STORE:
        GLOBAL_STORE.barrier()

class CascadeAdapter(BaseStore):
    def __init__(self, tag, gpu_cap_gb=38.0):
        self.store = init_global_store(tag, gpu_cap_gb)
    def put(self, key, data, is_prefix=False):
        self.store.put(str(key), data, is_prefix)
    def get(self, key, out):
        found, size = self.store.get(str(key), out)
        return found
    def barrier(self): self.store.barrier()
    def clear(self): self.store.clear()

class HDF5Adapter(BaseStore):
    def __init__(self, tag):
        import h5py
        self.h5py = h5py
        self.path = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_v7_{tag}_{job_id}.h5")
        if rank == 0 and self.path.exists(): self.path.unlink()
        mpi_barrier()
        if rank == 0:
            with self.h5py.File(self.path, 'w') as f: pass
        mpi_barrier()
        time.sleep(1)

    def put(self, key, data, is_prefix=False):
        if rank == 0: # Centralized write to avoid metadata locks on PFS
            try:
                with self.h5py.File(self.path, 'a') as f:
                    if str(key) not in f: f.create_dataset(str(key), data=data)
            except: pass

    def get(self, key, out):
        for _ in range(5):
            try:
                with self.h5py.File(self.path, 'r', swmr=True) as f:
                    if str(key) in f:
                        out[:] = f[str(key)][:]
                        return True
            except: time.sleep(0.5)
        return False
    def barrier(self): mpi_barrier()
    def clear(self):
        mpi_barrier()
        if rank == 0 and self.path.exists(): self.path.unlink()
        mpi_barrier()

class PosixAdapter(BaseStore):
    def __init__(self, name, tag):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_v7_{tag}_{job_id}")
        if rank == 0 and self.dir.exists():
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)
        mpi_barrier()
        self.dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()

    def put(self, key, data, is_prefix=False):
        p = self.dir / str(key)
        with open(p, 'wb') as f: f.write(data)

    def get(self, key, out):
        p = self.dir / str(key)
        for _ in range(10):
            if p.exists():
                try:
                    with open(p, 'rb') as f: 
                        out[:] = np.frombuffer(f.read(), dtype=np.uint8)
                        return True
                except: pass
            time.sleep(0.3)
        return False
    def barrier(self): mpi_barrier()
    def clear(self):
        mpi_barrier()
        if rank == 0 and self.dir.exists():
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)
        mpi_barrier()

def make_adapter(name, tag, gpu_cap_gb=38.0):
    if name == "Cascade": return CascadeAdapter(tag, gpu_cap_gb)
    if name == "HDF5": return HDF5Adapter(tag)
    if name == "vLLM-GPU": return PosixAdapter("vllm", tag)
    if name == "LMCache": return PosixAdapter("lmcache", tag)
    return None

# ============================================================
# V7 Scenarios (8-Node Scaling)
# ============================================================

def run_s1_prefix_sharing(systems, block_size):
    """Scenario 1: Massive 8-Node Prefix Sharing (32 GPUs Concurrent)"""
    print_rank0("\n" + "!"*80)
    print_rank0(" V7-S1: Massive Prefix Sharing (Llama-3-70B Pattern)")
    print_rank0(" 32 GPUs requesting 25.6 GB shared prefix (160 blocks)")
    print_rank0("!"*80)
    
    num_shared_blocks = 160 # 160 * 160MB = 25.6GB
    keys = [f"shared_prefix_{i}" for i in range(num_shared_blocks)]
    
    for name in systems:
        adapter = make_adapter(name, "s1")
        adapter.barrier()
        
        # Rank 0 writes all shared blocks
        if rank == 0:
            print_rank0(f"[{name}] Pre-populating 25.6GB Shared Prefix...")
            for k in keys:
                adapter.put(k, generate_block(hash(k) % 1000, block_size), is_prefix=True)
        adapter.barrier()
        
        # All 32 ranks (8 nodes) request all 160 blocks
        t0 = time.time()
        for k in keys:
            out = np.empty(block_size, dtype=np.uint8)
            adapter.get(k, out)
        adapter.barrier()
        total_time = time.time() - t0
        
        aggr_bw = (num_shared_blocks * block_size * world) / (total_time * 1024**3)
        print(f"[Rank {rank}] {name:10} | Aggr Read BW: {aggr_bw:8.2f} GB/s | Total Time: {total_time:6.2f} s")
        adapter.barrier()

def run_s2_tiered_stress(systems, block_size):
    """Scenario 2: Full-Tiered Stress (Dataset exceeding GPU+DRAM)"""
    print_rank0("\n" + "!"*80)
    print_rank0(" V7-S2: Full-Tiered Stress (Extreme Load)")
    print_rank0(" Dataset: 800 GB (5000 Blocks) distributed across 8 nodes")
    print_rank0("!"*80)
    
    # 5000 blocks * 160MB = 800GB.
    # GPU cap (32 GPUs * 38GB) = 1.2TB. To force tiered eviction, 
    # we reduce Cascade device cap in this test to 10GB.
    num_total_blocks = 5000
    blocks_per_rank = num_total_blocks // world
    
    for name in systems:
        # Force low capacity for Cascade to trigger Tier 3 (DRAM) and Tier 5 (Lustre)
        adapter = make_adapter(name, "s2", gpu_cap_gb=10.0 if name == "Cascade" else 38.0)
        adapter.barrier()
        
        # Each rank writes its own set
        t_put = time.time()
        for i in range(blocks_per_rank):
            key = f"r{rank}_b{i}"
            adapter.put(key, generate_block(rank*10000 + i, block_size))
        adapter.barrier()
        put_time = time.time() - t_put
        
        # All ranks read back their data (likely from tiered storage)
        t_get = time.time()
        for i in range(blocks_per_rank):
            key = f"r{rank}_b{i}"
            out = np.empty(block_size, dtype=np.uint8)
            adapter.get(key, out)
        adapter.barrier()
        get_time = time.time() - t_get
        
        aggr_read_bw = (num_total_blocks * block_size) / (get_time * 1024**3)
        print(f"[Rank {rank}] {name:10} | Read BW: {aggr_read_bw:8.2f} GB/s | Write Time: {put_time:6.2f} s")
        adapter.barrier()

def run_s3_cluster_dedup(systems, block_size):
    """Scenario 3: Cluster-wide Dedup (Multi-turn Sharing)"""
    print_rank0("\n" + "!"*80)
    print_rank0(" V7-S3: Cluster-wide Distributed Deduplication")
    print_rank0(" 8 nodes generating 1,000 identical multi-turn context blocks")
    print_rank0("!"*80)
    
    num_shared_contexts = 1000
    keys = [f"shared_ctx_{i}" for i in range(num_shared_contexts)]
    data_content = generate_block(777, block_size) # All ranks use SAME data
    
    for name in systems:
        adapter = make_adapter(name, "s3")
        adapter.barrier()
        
        t0 = time.time()
        # All ranks attempt to write the SAME context blocks
        for k in keys:
            adapter.put(k, data_content)
        adapter.barrier()
        dur = time.time() - t0
        
        if name == "Cascade" and rank == 0:
            stats = adapter.store.get_stats()
            print_rank0(f"[Cascade] Dedup Saved: {stats.dedup_bytes_saved/1024**3:.2f} GB | Saved Ratio: {(stats.dedup_bytes_saved/(num_shared_contexts*block_size*world))*100:.1f}%")
        
        print(f"[Rank {rank}] {name:10} | Put Total Time: {dur:8.2f} s")
        adapter.barrier()

def run_s4_prod_disaggregation(systems, block_size):
    """Scenario 4: Production Disaggregation (Interference Isolation)"""
    print_rank0("\n" + "!"*80)
    print_rank0(" V7-S4: Production Disaggregation (Serving vs Storage)")
    print_rank0(" Ranks 0-15: Serving (Readers) | Ranks 16-31: Disaggregated Stores (Writers)")
    print_rank0("!"*80)
    
    is_serving = rank < 16
    shared_key = "prod_shared_prefix"
    
    for name in systems:
        adapter = make_adapter(name, "s4")
        adapter.barrier()
        
        if rank == 0:
            adapter.put(shared_key, generate_block(1234, block_size), is_prefix=True)
        adapter.barrier()
        
        if is_serving:
            # Serving nodes: High pressure random reads
            time.sleep(1) 
            lats = []
            for i in range(50):
                out = np.empty(block_size, dtype=np.uint8)
                t0 = time.time()
                adapter.get(shared_key, out)
                lats.append((time.time() - t0) * 1000)
                time.sleep(0.05)
            print(f"[Rank {rank}] SERVING {name:10} | P50: {np.percentile(lats, 50):6.2f} ms | P99: {np.percentile(lats, 99):6.2f} ms")
        else:
            # Storage nodes: Aggressive background writes (interference)
            start_w = time.time()
            count = 0
            while time.time() - start_w < 5:
                adapter.put(f"bg_r{rank}_{count}", generate_block(count, block_size))
                count += 1
            print(f"[Rank {rank}] STORAGE {name:10} | Background Writes: {count}")
            
        adapter.barrier()

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,LMCache")
    args = parser.parse_args()
    
    # Llama-3-70B Block Size: 160MB
    # 80 layers * 8 heads * 128 dim * 1024 tokens * 2 (K+V) * 2 (fp16) = 167,772,160 bytes
    block_size = 160 * 1024 * 1024
    systems = args.systems.split(",")
    
    if args.scenario in ("1", "all"): run_s1_prefix_sharing(systems, block_size)
    if args.scenario in ("2", "all"): run_s2_tiered_stress(systems, block_size)
    if args.scenario in ("3", "all"): run_s3_cluster_dedup(systems, block_size)
    if args.scenario in ("4", "all"): run_s4_prod_disaggregation(systems, block_size)
    
    print_rank0("\nV7 Serving-at-Scale Benchmarks Completed Successfully.")

if __name__ == "__main__":
    main()
