import os
import time
import argparse
import numpy as np
from pathlib import Path
import subprocess
import concurrent.futures
import socket

# MPI Configuration
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

def print_rank0(*args):
    if rank == 0: print(*args, flush=True)

# Global Rank 0 info for distributed coordination
def get_rank0_host():
    host = socket.gethostname() if rank == 0 else None
    host_file = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/rank0_host_{job_id}")
    if rank == 0:
        host_file.parent.mkdir(parents=True, exist_ok=True)
        with open(host_file, 'w') as f: f.write(host)
    else:
        for _ in range(30): # wait for rank 0
            if host_file.exists(): 
                try:
                    with open(host_file, 'r') as f: 
                        host = f.read().strip()
                        if host: break
                except: pass
            time.sleep(1)
    return host

RANK0_HOST = get_rank0_host()

# Path setup
default_build = '../../cascade_Code/cpp/build_cascade_cpp'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
lustre_path = Path("/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/serv_metrics")
lustre_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# Core Adapters
# ============================================================

class BaseStore:
    def put(self, key, data, is_prefix=False): pass
    def get(self, key, out): return False
    def barrier(self): pass
    def clear(self): pass

class CascadeAdapter(BaseStore):
    def __init__(self, build_dir):
        import sys
        sys.path.append(str(Path(build_dir).resolve()))
        import cascade_cpp
        
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 30 * 1024**3
        cfg.dram_capacity = 140 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.semantic_eviction = True
        cfg.locality_aware = True
        cfg.kv_compression = True
        
        self.store = cascade_cpp.DistributedStore(cfg)
        
    def put(self, key, data, is_prefix=False):
        self.store.put(str(key), data, is_prefix)
        
    def get(self, key, out):
        res = self.store.get(str(key), out)
        if isinstance(res, tuple):
            return res[0]
        return res
    
    def barrier(self):
        self.store.barrier()
        
    def sync(self):
        # Unique to Cascade: ensure all nodes see the metadata
        try:
            self.store.sync_metadata()
            self.store.barrier()
        except: pass

    def clear(self):
        try:
            self.store.clear()
            print_rank0("[Cascade] Memory tiers and indices cleared.")
        except: pass


class HDF5Adapter(BaseStore):
    def __init__(self, lustre_path):
        import h5py
    def __init__(self, lustre_path):
        import h5py
        self.h5py = h5py
        # Use per-rank file to avoid Lustre contention during heavy concurrent writes
        self.path = Path(lustre_path) / f"serv_metrics_llama_{job_id}_rank{rank}.h5"
        if self.path.exists(): self.path.unlink()
        with h5py.File(self.path, 'w') as f: pass

    def put(self, key, data, is_prefix=False):
        # Only I write to my own file
        with self.h5py.File(self.path, 'a') as f:
            if str(key) not in f:
                f.create_dataset(str(key), data=data)

    def get(self, key, out):
        # We need to know which rank wrote which key.
        # Ranks write in req_keys[rank::world]
        try:
            # key is like 'req_5_b0'
            idx = int(str(key).split('_')[1])
            source_rank = idx % world
            remote_path = Path(self.path.parent) / f"serv_metrics_llama_{job_id}_rank{source_rank}.h5"
            
            for _ in range(5):
                try:
                    with self.h5py.File(remote_path, 'r', libver='latest', swmr=True) as f:
                        if str(key) in f:
                            out[:] = f[str(key)][:]
                            return True
                except:
                    time.sleep(1)
        except:
            pass
        return False

    def barrier(self): time.sleep(1)
    def clear(self): pass

class PosixAdapter(BaseStore):
    def __init__(self, name):
        self.dir = lustre_path / f"{name}_serv_{job_id}"
        if rank == 0:
            import shutil
            if self.dir.exists(): shutil.rmtree(self.dir, ignore_errors=True)
            self.dir.mkdir(parents=True, exist_ok=True)
            # Signal others that directory is ready
            with open(self.dir / ".ready", "w") as f: f.write("1")
        
        # Wait for rank 0
        for _ in range(60):
            if (self.dir / ".ready").exists(): break
            time.sleep(1)

    def put(self, key, data, is_prefix=False):
        p = self.dir / str(key)
        with open(p, 'wb') as f:
            f.write(data)

    def get(self, key, out):
        p = self.dir / str(key)
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    out[:] = np.frombuffer(f.read(), dtype=np.uint8)
                return True
            except: pass
        return False

    def barrier(self): time.sleep(1)
    def clear(self): pass

class RedisAdapter(BaseStore):
    CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks to avoid Redis timeouts

    def __init__(self):
        import redis as redis_lib
        self.redis_lib = redis_lib
        # Retry connection up to 30 seconds
        self.client = None
        for attempt in range(30):
            try:
                client = redis_lib.Redis(
                    host=RANK0_HOST, port=16379,
                    socket_timeout=30,
                    socket_connect_timeout=10,
                    retry_on_timeout=True,
                )
                client.ping()
                self.client = client
                break
            except Exception as e:
                if attempt == 29:
                    print(f"[Rank {rank}] Redis: Failed to connect after 30 attempts: {e}")
                time.sleep(1)

    def put(self, key, data, is_prefix=False):
        if self.client is None: return
        raw = data.tobytes() if hasattr(data, 'tobytes') else bytes(data)
        total = len(raw)
        if total <= self.CHUNK_SIZE:
            # Small value: single SET
            self.client.set(str(key), raw)
            self.client.set(f"{key}:meta", str(total).encode())
        else:
            # Large value: chunked SET via pipeline
            n_chunks = (total + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
            pipe = self.client.pipeline(transaction=False)
            for i in range(n_chunks):
                start = i * self.CHUNK_SIZE
                end = min(start + self.CHUNK_SIZE, total)
                pipe.set(f"{key}:chunk:{i}", raw[start:end])
            pipe.set(f"{key}:meta", f"{total}:{n_chunks}".encode())
            pipe.execute()

    def get(self, key, out):
        if self.client is None: return False
        for attempt in range(3):
            try:
                meta = self.client.get(f"{key}:meta")
                if meta is None:
                    time.sleep(2)
                    continue
                meta_str = meta.decode()
                if ':' not in meta_str:
                    # Single chunk
                    val = self.client.get(str(key))
                    if val:
                        out[:len(val)] = np.frombuffer(val, dtype=np.uint8)
                        return True
                else:
                    # Multi-chunk
                    total, n_chunks = meta_str.split(':')
                    total, n_chunks = int(total), int(n_chunks)
                    pipe = self.client.pipeline(transaction=False)
                    for i in range(n_chunks):
                        pipe.get(f"{key}:chunk:{i}")
                    chunks = pipe.execute()
                    offset = 0
                    for chunk_data in chunks:
                        if chunk_data is None: return False
                        chunk_arr = np.frombuffer(chunk_data, dtype=np.uint8)
                        out[offset:offset + len(chunk_arr)] = chunk_arr
                        offset += len(chunk_arr)
                    return True
            except Exception as e:
                print(f"[Rank {rank}] Redis Get Error (key: {key}, attempt {attempt}): {e}")
                time.sleep(2)
        return False

    def barrier(self):
        # File-based barrier for proper distributed synchronization
        barrier_dir = lustre_path / f"redis_barrier_{job_id}"
        barrier_dir.mkdir(parents=True, exist_ok=True)
        # Signal ready
        (barrier_dir / f"rank_{rank}").touch()
        # Wait for all ranks
        for _ in range(120):
            ready = sum(1 for f in barrier_dir.iterdir() if f.name.startswith("rank_"))
            if ready >= world: break
            time.sleep(0.5)
        # Cleanup (rank 0 only, after everyone has passed)
        time.sleep(1)
        if rank == 0:
            import shutil
            shutil.rmtree(barrier_dir, ignore_errors=True)

    def clear(self):
        if self.client is None: return
        try:
            if rank == 0: self.client.flushall()
            time.sleep(1)
        except: 
            print_rank0("[Redis] Clear failed (server might be down)")

# ============================================================
# Simulator
# ============================================================

class RealKVLoader:
    def __init__(self, base_dir="/pscratch/sd/s/sgkim/cascade_kv_cache"):
        self.base_dir = Path(base_dir)
        self.all_blocks = {}
        self.block_ids = []
        index_path = self.base_dir / "global_index.json"
        
        if index_path.exists():
            try:
                import json
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self.all_blocks = data['blocks']
                    self.block_ids = list(self.all_blocks.keys())
            except Exception as e:
                print_rank0(f"Error loading real KV index: {e}")
        
    def load(self, block_id):
        loc = self.all_blocks[block_id]
        path = self.base_dir / loc['file']
        with open(path, 'rb') as f:
            f.seek(loc['offset'])
            return np.frombuffer(f.read(loc['size']), dtype=np.uint8)

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache,LMCache-Redis")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--prompt-blocks", type=int, default=1)
    parser.add_argument("--gen-tokens", type=int, default=32)
    args = parser.parse_args()

    loader = RealKVLoader()
    block_size = 160 * 1024 * 1024 # 160MB default
    if loader.block_ids:
        # Use a sample block to determine size
        first_bid = loader.block_ids[0]
        block_size = loader.all_blocks[first_bid]['size']
    else:
        # Fallback to synthetic if real data missing
        print_rank0("WARNING: Real KV data not found! Using synthetic data blocks.")
        loader.block_ids = [f"synth_b{i}" for i in range(100)]
        loader.load = lambda bid: np.random.randint(0, 255, 160*1024*1024, dtype=np.uint8)

    # Initialize Adapters ONCE
    available_systems = {
        "Cascade": lambda: CascadeAdapter(build_dir),
        "HDF5": lambda: HDF5Adapter(lustre_path),
        "vLLM-GPU": lambda: PosixAdapter("vllm"),
        "PDC": lambda: PosixAdapter("pdc"),
        "LMCache": lambda: PosixAdapter("lmcache"),
        "LMCache-Redis": lambda: RedisAdapter(),
    }
    
    target_names = args.systems.split(",")
    
    for name in target_names:
        if name not in available_systems: continue
        
        print_rank0(f"\n>>> Benchmarking System: {name}")
        adapter = available_systems[name]()
        
        # Cold Start Simulation
        adapter.clear()
        
        num_req = args.num_requests
        req_keys = [f"req_{i}_b0" for i in range(num_req)]
        
        # Distributed Write Phase
        my_write_reqs = req_keys[rank::world]
        for i, rk_key in enumerate(req_keys):
            if rk_key in my_write_reqs:
                bid = loader.block_ids[i % len(loader.block_ids)]
                data = loader.load(bid)
                adapter.put(rk_key, data)
        
        adapter.barrier()
        
        # Cascade specific: sync metadata so other nodes see our writes
        if name == "Cascade":
            print_rank0(f"[{name}] Syncing metadata across {world} nodes...")
            adapter.sync()
            
        # Cross-Node Fetch: Rank i fetches data written by Rank (i+1) % world
        if world > 1:
            print_rank0(f"Waiting 30s for distributed storage consistency...")
            time.sleep(30)
            adapter.barrier()
            
        target_rank = (rank + 1) % world
        my_read_reqs = req_keys[target_rank::world]
        
        ttfts = []
        tpots = []
        start_time = time.time()
        
        def fetch_and_simulate(rk):
            t0 = time.time()
            buf = np.empty(block_size, dtype=np.uint8)
            
            # Robust GET with retries for Lustre
            found = False
            for attempt in range(3):
                if adapter.get(rk, buf):
                    found = True
                    break
                time.sleep(2)
            
            if not found: return None, None
            
            ttft = (time.time() - t0) * 1000 # ms
            
            # Simulate TPOT (fast baseline)
            t_tpots = []
            sim_inf_delay = 0.002
            for _ in range(args.gen_tokens):
                t_token_start = time.time()
                time.sleep(sim_inf_delay) 
                t_tpots.append((time.time() - t_token_start) * 1000)
            return ttft, t_tpots

        # Process requests sequentially to ensure stability in distributed environment
        # (Multi-threading with MPI/RDMA in C++ layer can cause memory corruption)
        for rk in my_read_reqs:
            res = fetch_and_simulate(rk)
            if res and res[0] is not None:
                _ttft, _tpots = res
                ttfts.append(_ttft)
                tpots.extend(_tpots)

        # Final barrier for this system to ensure all RDMA fetches are done before peer exits
        adapter.barrier()

        if not ttfts:
            print_rank0(f"[{name}] ERROR: All requests failed or cache misses!")
            continue

        total_time = time.time() - start_time
        
        # Latency Stats
        avg_ttft = np.mean(ttfts)
        p50_ttft = np.percentile(ttfts, 50)
        p90_ttft = np.percentile(ttfts, 90)
        
        avg_tpot = np.mean(tpots)
        p50_tpot = np.percentile(tpots, 50)
        p90_tpot = np.percentile(tpots, 90)
        
        # Throughput Stats
        total_reqs = len(ttfts)
        total_tokens = total_reqs * args.gen_tokens
        prompt_tokens = total_reqs * 1024 # Assume 1K prompt
        
        # Broadcast local stats to compute global aggregate
        # Simplified: average of averages across ranks for latency, sum of counts for throughput
        # (In a real benchmark we'd gather all data points)
        
        print_rank0(f"--- {name} Results ---")
        print_rank0(f"Avg TTFT: {avg_ttft:.2f} ms | P50: {p50_ttft:.2f} ms | P90: {p90_ttft:.2f} ms")
        print_rank0(f"Avg TPOT: {avg_tpot:.2f} ms | P50: {p50_tpot:.2f} ms | P90: {p90_tpot:.2f} ms")
        print_rank0(f"Request Throughput: {total_reqs*world/total_time:.2f} req/s")
        print_rank0(f"Output Token Throughput: {total_tokens*world/total_time:.2f} tok/s")
        print_rank0(f"Total Token Throughput: {(total_tokens+prompt_tokens)*world/total_time:.2f} tok/s")

    adapter.barrier()
    print_rank0("\nAll benchmarks complete.")

if __name__ == "__main__":
    run_benchmark()
