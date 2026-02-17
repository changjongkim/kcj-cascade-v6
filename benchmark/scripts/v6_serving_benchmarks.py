#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import argparse
from pathlib import Path

# MPI Configuration (using SLURM env vars for rank info)
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

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

# ============================================================
# Helpers
# ============================================================

def generate_block(seed, size):
    np.random.seed(seed)
    return np.random.randint(0, 255, size, dtype=np.uint8)

# ============================================================
# Storage Adapters
# ============================================================

class BaseStore:
    def put(self, key, data, is_prefix=False): pass
    def get(self, key, out): pass
    def barrier(self): pass
    def cleanup(self): pass
    def clear(self): pass

GLOBAL_STORE = None

def init_global_store(tag="serving", gpu_cap_gb=38.0):
    global GLOBAL_STORE
    if GLOBAL_STORE is None:
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = int(gpu_cap_gb * 1024**3)
        cfg.dram_capacity = 40 * 1024**3 # intentionally small for eviction tests
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.semantic_eviction = True
        cfg.locality_aware = True
        cfg.promotion_threshold = 3
        cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_serv_{tag}_{job_id}"
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
    def cleanup(self): pass

class HDF5Adapter(BaseStore):
    def __init__(self, tag):
        import h5py
        self.h5py = h5py
        self.path = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_serv_{tag}_{job_id}.h5")
        if rank == 0 and self.path.exists(): self.path.unlink()
        mpi_barrier()
        # Ensure file exists for SWMR
        if rank == 0:
            with self.h5py.File(self.path, 'w') as f: pass
        mpi_barrier()
        time.sleep(1)

    def put(self, key, data, is_prefix=False):
        # HDF5 is strictly single-writer
        if rank == 0:
            try:
                with self.h5py.File(self.path, 'a') as f:
                    if str(key) not in f: f.create_dataset(str(key), data=data)
            except Exception as e:
                print(f"[Rank 0] HDF5 Put Error: {e}")

    def get(self, key, out):
        # Retry logic for Lustre metadata/locking
        for _ in range(3):
            try:
                with self.h5py.File(self.path, 'r', swmr=True) as f:
                    if str(key) in f:
                        out[:] = f[str(key)][:]
                        return True
            except:
                time.sleep(1)
        return False
    def barrier(self): mpi_barrier()
    def clear(self):
        mpi_barrier()
        if rank == 0 and self.path.exists():
            try: self.path.unlink()
            except: pass
        mpi_barrier()

class PosixAdapter(BaseStore):
    def __init__(self, name, tag):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_serv_{tag}_{job_id}")
        if rank == 0 and self.dir.exists():
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)
        mpi_barrier()
        # All ranks attempt to create to ensure visibility on all nodes immediately
        self.dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()
        time.sleep(1)

    def put(self, key, data, is_prefix=False):
        # Multi-node shared writes in Posix are risky, usually Rank 0 handles shared
        p = self.dir / str(key)
        with open(p, 'wb') as f: f.write(data)

    def get(self, key, out):
        p = self.dir / str(key)
        for _ in range(5):
            if p.exists():
                try:
                    with open(p, 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
                    return True
                except: pass
            time.sleep(0.5)
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
    if name == "PDC": return PosixAdapter("pdc", tag)
    if name == "LMCache": return PosixAdapter("lmcache", tag)
    return None

# ============================================================
# Scenario 1: Hot-spot Mitigation (Locality)
# ============================================================
def run_scenario1(systems, block_size):
    print_rank0("\n" + "="*80)
    print_rank0(" SCENARIO 1: Hot-spot Mitigation (Locality-Aware Promotion)")
    print_rank0("="*80)
    
    key = "shared_hot_block"
    data = generate_block(42, block_size)
    
    for name in systems:
        adapter = make_adapter(name, "s1")
        adapter.clear()
        
        # Rank 0 writes the block
        if rank == 0:
            adapter.put(key, data)
        adapter.barrier()
        
        # Ranks 1-3 fetch it 10 times
        latencies = []
        if rank > 0:
            for i in range(10):
                out = np.empty(block_size, dtype=np.uint8)
                t0 = time.time()
                adapter.get(key, out)
                latencies.append((time.time() - t0) * 1000)
                # Sleep slightly to avoid overlapping with MPI internal state too much
                time.sleep(0.1)
        
        adapter.barrier()
        if rank > 0:
            avg_first_3 = np.mean(latencies[:3])
            avg_last_3 = np.mean(latencies[-3:])
            reduction = avg_first_3 / avg_last_3 if avg_last_3 > 0 else 1
            print(f"[Rank {rank}] {name:10} | Initial Lat: {avg_first_3:8.2f} ms | Final Lat: {avg_last_3:8.2f} ms | Speedup: {reduction:.1f}x")
        
        adapter.barrier()

# ============================================================
# Scenario 2: Semantic Eviction
# ============================================================
def run_scenario2(systems, block_size):
    print_rank0("\n" + "="*80)
    print_rank0(" SCENARIO 2: Semantic Eviction (Prefix Protection)")
    print_rank0("="*80)
    
    # Intentionally small blocks to trigger eviction quickly
    # We will use 0.5 GB GPU capacity for Cascade to force eviction
    num_prefix = 2
    num_normal = 8
    
    for name in systems:
        # Use small GPU cap for Cascade
        adapter = make_adapter(name, "s2", gpu_cap_gb=0.5 if name == "Cascade" else 38.0)
        adapter.clear()
        
        prefix_keys = [f"prefix_{i}" for i in range(num_prefix)]
        normal_keys = [f"normal_{i}" for i in range(num_normal)]
        data = generate_block(rank, block_size)
        
        # 1. Put prefix blocks (protected)
        for k in prefix_keys:
            adapter.put(k, data, is_prefix=True)
        
        # 2. Put normal blocks to fill cache and trigger eviction
        for k in normal_keys:
            adapter.put(k, data, is_prefix=False)
        
        adapter.barrier()
        
        # 3. Verify if prefix blocks are still "Fast" (in Cascade) or "Exist" (in baselines)
        # Actually for baselines we check if they are still there.
        prefix_hits = 0
        for k in prefix_keys:
            out = np.empty(block_size, dtype=np.uint8)
            t0 = time.time()
            found = adapter.get(k, out)
            lat = (time.time() - t0) * 1000
            if found:
                prefix_hits += 1
                if name == "Cascade":
                    status = "Hot (Protected)" if lat < 5 else "Cold (Evicted)"
                    print(f"[Rank {rank}] {name} | {k} | Lat: {lat:6.2f} ms | {status}")
        
        print(f"[Rank {rank}] {name:10} | Prefix Retention: {prefix_hits}/{num_prefix}")
        adapter.barrier()

# ============================================================
# Scenario 3: Distributed Dedup
# ============================================================
def run_scenario3(systems, block_size):
    print_rank0("\n" + "="*80)
    print_rank0(" SCENARIO 3: Context-Sharing (Distributed Deduplication)")
    print_rank0("="*80)
    
    key = "global_dedup_block"
    data = generate_block(99, block_size) # All ranks use same seed -> same data
    
    for name in systems:
        adapter = make_adapter(name, "s3")
        adapter.clear()
        
        t0 = time.time()
        adapter.put(key, data)
        adapter.barrier()
        dur = (time.time() - t0) * 1000
        
        # In Cascade, rank 0 will write, ranks 1-3 should be near-instant
        print(f"[Rank {rank}] {name:10} | Put Latency: {dur:8.2f} ms")
        
        if name == "Cascade" and rank == 0:
            stats = adapter.store.get_stats()
            print(f"[Rank 0] Cascade Dedup Hits: {stats.dedup_hits} | Saved: {stats.dedup_bytes_saved/1024**2:.1f} MB")
            
        adapter.barrier()

# ============================================================
# Scenario 4: Disaggregated Stress
# ============================================================
def run_scenario4(systems, block_size):
    print_rank0("\n" + "="*80)
    print_rank0(" SCENARIO 4: Disaggregated Stress (interference)")
    print_rank0("="*80)
    
    # Ranks 0-1: Writers (stressing backend)
    # Ranks 2-3: Readers (measuring tail performance)
    
    for name in systems:
        adapter = make_adapter(name, "s4")
        adapter.clear()
        
        # Pre-populate some data for readers
        shared_key = "stress_shared_block"
        if rank == 0:
            adapter.put(shared_key, generate_block(123, block_size))
        adapter.barrier()
        
        stop_flag = False
        if rank < 2:
            # Writers
            start_w = time.time()
            count = 0
            while time.time() - start_w < 5:
                adapter.put(f"stress_r{rank}_{count}", generate_block(count, block_size))
                count += 1
            print(f"[Rank {rank}] Writer {name} | Completed {count} writes")
        else:
            # Readers
            time.sleep(1) # wait for writers to start
            lats = []
            for i in range(20):
                out = np.empty(block_size, dtype=np.uint8)
                t0 = time.time()
                adapter.get(shared_key, out)
                lats.append((time.time() - t0) * 1000)
                time.sleep(0.1)
            print(f"[Rank {rank}] Reader {name} | Avg Lat: {np.mean(lats):.2f} ms | P99: {np.percentile(lats, 99):.2f} ms")
            
        adapter.barrier()

# ============================================================
# Scenario 5: Tail Latency
# ============================================================
def run_scenario5(systems, block_size):
    print_rank0("\n" + "="*80)
    print_rank0(" SCENARIO 5: Tail Latency Analysis (P99 Jitter)")
    print_rank0("="*80)
    
    num_ops = 100
    
    for name in systems:
        adapter = make_adapter(name, "s5")
        adapter.clear()
        
        # Pre-load 100 blocks
        keys = [f"tail_block_{i}" for i in range(num_ops)]
        if rank == 0:
            for k in keys:
                adapter.put(k, generate_block(abs(hash(k)) % 1000, block_size))
        adapter.barrier()
        
        # Random reads
        lats = []
        rng = np.random.RandomState(rank)
        for i in range(num_ops):
            k = keys[rng.randint(0, num_ops)]
            out = np.empty(block_size, dtype=np.uint8)
            t0 = time.time()
            adapter.get(k, out)
            lats.append((time.time() - t0) * 1000)
            
        lats = np.array(lats)
        print(f"[Rank {rank}] {name:10} | P50: {np.percentile(lats, 50):6.2f} ms | P95: {np.percentile(lats, 95):6.2f} ms | P99: {np.percentile(lats, 99):6.2f} ms")
        adapter.barrier()

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["1", "2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache")
    parser.add_argument("--block-size-mb", type=int, default=320)
    args = parser.parse_args()
    
    systems = args.systems.split(",")
    block_size = args.block_size_mb * 1024 * 1024
    
    if args.scenario in ("1", "all"): run_scenario1(systems, block_size)
    if args.scenario in ("2", "all"): run_scenario2(systems, block_size)
    if args.scenario in ("3", "all"): run_scenario3(systems, block_size)
    if args.scenario in ("4", "all"): run_scenario4(systems, block_size)
    if args.scenario in ("5", "all"): run_scenario5(systems, block_size)
    
    print_rank0("\nServing benchmarks completed.")

if __name__ == "__main__":
    main()
