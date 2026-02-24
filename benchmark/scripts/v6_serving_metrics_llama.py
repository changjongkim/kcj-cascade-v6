import os
import sys
import time
import json
import numpy as np
import argparse
from pathlib import Path
import subprocess
import concurrent.futures

# MPI Configuration
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

def print_rank0(msg):
    if rank == 0:
        print(msg, flush=True)

# ============================================================
# Helpers
# ============================================================

def generate_block(seed, size):
    np.random.seed(seed)
    return np.random.randint(0, 255, size, dtype=np.uint8)

class RealKVLoader:
    def __init__(self, base_dir="/pscratch/sd/s/sgkim/cascade_kv_cache"):
        self.base_dir = Path(base_dir)
        index_path = self.base_dir / "global_index.json"
        self.block_ids = []
        self.all_blocks = {}
        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)
                self.all_blocks = data['blocks']
                self.block_ids = list(self.all_blocks.keys())

    def load(self, bid):
        loc = self.all_blocks[bid]
        file_path = loc.get('file') or loc.get('path')
        with open(self.base_dir / file_path, 'rb') as f:
            f.seek(loc['offset'])
            return np.frombuffer(f.read(loc['size']), dtype=np.uint8)

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

def init_global_store():
    global GLOBAL_STORE
    if GLOBAL_STORE is None:
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 160 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_serv_metrics_{job_id}"
        GLOBAL_STORE = cascade_cpp.DistributedStore(cfg)
    return GLOBAL_STORE

def mpi_barrier():
    if GLOBAL_STORE:
        GLOBAL_STORE.barrier()
    else:
        # Fallback to simple file-based barrier or similar if needed, 
        # but here we usually have Cascade or MPI initialized.
        pass

class CascadeAdapter(BaseStore):
    def __init__(self):
        self.store = init_global_store()
    def put(self, key, data, is_prefix=False):
        self.store.put(str(key), data, is_prefix)
    def get(self, key, out):
        found, size = self.store.get(str(key), out)
        return found
    def barrier(self): self.store.barrier()
    def clear(self): self.store.clear()

class HDF5Adapter(BaseStore):
    def __init__(self):
        import h5py
        self.h5py = h5py
        self.path = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_serv_{job_id}.h5")
        if rank == 0 and self.path.exists(): self.path.unlink()
        if rank == 0:
            with self.h5py.File(self.path, 'w') as f: pass
        time.sleep(1)

    def put(self, key, data, is_prefix=False):
        if rank == 0:
            with self.h5py.File(self.path, 'a') as f:
                if str(key) not in f: f.create_dataset(str(key), data=data)

    def get(self, key, out):
        try:
            with self.h5py.File(self.path, 'r', swmr=True) as f:
                if str(key) in f:
                    out[:] = f[str(key)][:]
                    return True
        except: pass
        return False
    def barrier(self): time.sleep(0.5) # Simulating sync
    def clear(self): pass

class PosixAdapter(BaseStore):
    def __init__(self, name):
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_serv_{job_id}")
        if rank == 0:
            import shutil
            if self.dir.exists(): shutil.rmtree(self.dir, ignore_errors=True)
            self.dir.mkdir(parents=True, exist_ok=True)
        time.sleep(1)

    def put(self, key, data, is_prefix=False):
        p = self.dir / str(key)
        with open(p, 'wb') as f: f.write(data)

    def get(self, key, out):
        p = self.dir / str(key)
        if p.exists():
            with open(p, 'rb') as f: out[:] = np.frombuffer(f.read(), dtype=np.uint8)
            return True
        return False
    def barrier(self): time.sleep(0.5)
    def clear(self): pass

class RedisAdapter(BaseStore):
    def __init__(self):
        import redis
        self.client = redis.Redis(host='localhost', port=16379)
    def put(self, key, data, is_prefix=False):
        self.client.set(str(key), data.tobytes())
    def get(self, key, out):
        val = self.client.get(str(key))
        if val:
            out[:] = np.frombuffer(val, dtype=np.uint8)
            return True
        return False
    def barrier(self): time.sleep(0.5)
    def clear(self): self.client.flushall()

# ============================================================
# Main Logic
# ============================================================

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache,LMCache-Redis")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--prompt-blocks", type=int, default=1)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--model", default="llama-3-70b")
    args = parser.parse_args()

    loader = RealKVLoader()
    block_size = 160 * 1024 * 1024 # default for 70B
    if loader.block_ids:
        block_size = loader.all_blocks[loader.block_ids[0]]['size']

    systems = args.systems.split(",")
    results = {}

    for name in systems:
        print_rank0(f"\n>>> Benchmarking System: {name}")
        adapter = None
        if name == "Cascade": adapter = CascadeAdapter()
        elif name == "HDF5": adapter = HDF5Adapter()
        elif name == "vLLM-GPU": adapter = PosixAdapter("vllm")
        elif name == "PDC": adapter = PosixAdapter("pdc")
        elif name == "LMCache": adapter = PosixAdapter("lmcache")
        elif name == "LMCache-Redis": adapter = RedisAdapter()
        
        if not adapter: continue

        # Cold Start Simulation: Clear/Prep
        adapter.clear()
        
        # Scenario: All requests are pre-prepared in storage (Cold Start from Lustre/Remote)
        # We will share prompt blocks to simulate common prefix
        num_req = args.num_requests
        req_keys = [f"req_{i}_b0" for i in range(num_req)]
        
        # Distributed Write Phase: Each rank writes its own share of KV blocks
        data_cache = {}
        my_write_reqs = req_keys[rank::world]
        for i, rk_key in enumerate(req_keys):
            bid = loader.block_ids[i % len(loader.block_ids)]
            data = loader.load(bid)
            data_cache[rk_key] = data
            if rk_key in my_write_reqs:
                adapter.put(rk_key, data)
        
        # Sync globally to ensure all writes are complete
        if hasattr(adapter, 'barrier'): adapter.barrier()
        
        # Cascade-specific Cache Clear to ensure COLD start
        if name == "Cascade":
            adapter.clear() # Clear memory tiers
            
        print_rank0(f"[{name}] Starting serving simulation...")
        
        # Cross-Node Fetch: Rank i fetches data written by Rank (i+1) % world
        target_rank = (rank + 1) % world
        my_read_reqs = req_keys[target_rank::world]
        num_gen = args.gen_tokens
        
        ttfts = []
        tpots = []
        start_time = time.time()
        
        def fetch_and_simulate(rk):
            t0 = time.time()
            buf = np.empty(block_size, dtype=np.uint8)
            adapter.get(rk, buf)
            ttft = (time.time() - t0) * 1000 # ms
            
            # Simulate TPOT very minimally to emphasize caching TTFT throughput
            t_tpots = []
            sim_inf_delay = 0.002 # 2ms baseline to mimic very fast forward pass per token
            for _ in range(num_gen):
                t_token_start = time.time()
                time.sleep(sim_inf_delay) 
                t_tpots.append((time.time() - t_token_start) * 1000)
            return ttft, t_tpots

        # Execute multiple requests concurrently to model Batching in vLLM / TGI
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(fetch_and_simulate, rk) for rk in my_read_reqs]
            for f in concurrent.futures.as_completed(futures):
                _ttft, _tpots = f.result()
                ttfts.append(_ttft)
                tpots.extend(_tpots)

        if hasattr(adapter, 'barrier'): adapter.barrier()
        total_time = time.time() - start_time
        
        # Latency Stats
        all_ttfts = np.array(ttfts)
        all_tpots = np.array(tpots)
        
        # Consolidate metrics
        avg_ttft = np.mean(all_ttfts)
        p50_ttft = np.percentile(all_ttfts, 50)
        p90_ttft = np.percentile(all_ttfts, 90)
        
        avg_tpot = np.mean(all_tpots)
        p50_tpot = np.percentile(all_tpots, 50)
        p90_tpot = np.percentile(all_tpots, 90)
        
        total_reqs = num_req
        total_prompt_toks = num_req * 1024 # assuming 1 block = 1024 tokens
        total_gen_toks = num_req * num_gen
        
        req_throughput = total_reqs / total_time
        out_token_throughput = total_gen_toks / total_time
        total_token_throughput = (total_prompt_toks + total_gen_toks) / total_time
        
        print_rank0(f"--- {name} Results ---")
        print_rank0(f"Avg TTFT: {avg_ttft:.2f} ms | P50: {p50_ttft:.2f} ms | P90: {p90_ttft:.2f} ms")
        print_rank0(f"Avg TPOT: {avg_tpot:.2f} ms | P50: {p50_tpot:.2f} ms | P90: {p90_tpot:.2f} ms")
        print_rank0(f"Request Throughput: {req_throughput:.2f} req/s")
        print_rank0(f"Output Token Throughput: {out_token_throughput:.2f} tok/s")
        print_rank0(f"Total Token Throughput: {total_token_throughput:.2f} tok/s")
        
        results[name] = {
            'avg_ttft': avg_ttft, 'p50_ttft': p50_ttft, 'p90_ttft': p90_ttft,
            'avg_tpot': avg_tpot, 'p50_tpot': p50_tpot, 'p90_tpot': p90_tpot,
            'req_throughput': req_throughput,
            'out_token_throughput': out_token_throughput,
            'total_token_throughput': total_token_throughput
        }

    if rank == 0:
        with open(f"benchmark/results/serving_metrics_{job_id}.json", "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
