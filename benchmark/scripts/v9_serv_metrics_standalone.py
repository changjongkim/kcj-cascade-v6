import os
import time
import argparse
import numpy as np
from pathlib import Path
import subprocess
import socket
import sys
import json

# Add repo root to path for adapters
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

# Import adapter factory
from benchmark.run_benchmark import get_adapter

# MPI rank/world from SLURM
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

def print_rank0(*args):
    if rank == 0: 
        print(*args, flush=True)

# Helper for Lustre metadata sync
def force_lustre_sync(path):
    try:
        p = Path(path)
        if p.exists():
            subprocess.run(["ls", "-ld", str(p)], capture_output=True, timeout=5)
            if p.is_dir():
                subprocess.run(["ls", "-a", str(p)], capture_output=True, timeout=5)
        else:
            subprocess.run(["ls", "-ld", str(p.parent)], capture_output=True, timeout=5)
    except: pass

# Global Hostname coordination
def get_all_hosts(run_id=None):
    rid = run_id or job_id
    my_host = socket.gethostname()
    tmp_dir = REPO_ROOT / "benchmark" / "tmp" / f"hosts_{rid}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp_dir / f"rank_{rank}", 'w') as f:
        f.write(my_host)
    
    all_hosts = {}
    for _ in range(120):
        try:
            files = list(tmp_dir.iterdir())
            if len(files) >= world:
                for f in files:
                    try:
                        r_idx = int(f.name.split('_')[1])
                        with open(f, 'r') as hf:
                            all_hosts[r_idx] = hf.read().strip()
                    except: continue
                if len(all_hosts) >= world: break
        except: pass
        time.sleep(1)
    return all_hosts

# Barrier using filesystem (most independent way since we are splitting jobs)
def file_barrier(name, run_id=None):
    rid = run_id or job_id
    bar_dir = REPO_ROOT / "benchmark" / "tmp" / f"bar_{rid}_{name}"
    bar_dir.mkdir(parents=True, exist_ok=True)
    (bar_dir / f"rank_{rank}").touch()
    while True:
        try:
            if len(list(bar_dir.iterdir())) >= world: break
        except: pass
        time.sleep(1)
    time.sleep(2)

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
        if not self.all_blocks:
            import numpy as np
            # Synthetic fallback with some compressible pattern
            data = np.zeros(160*1024*1024, dtype=np.uint8).tobytes()
            mid = len(data) // 2
            return data[:mid], data[mid:]
            
        loc = self.all_blocks[block_id]
        path = self.base_dir / loc['file']
        with open(path, 'rb') as f:
            f.seek(loc['offset'])
            data = f.read(loc['size'])
            mid = len(data) // 2
            return data[:mid], data[mid:]

loader = RealKVLoader()
if getattr(loader, "block_ids", []):
    print_rank0(f"KV data loader initialized with {len(loader.block_ids)} real blocks.")
else:
    print_rank0("WARNING: Real KV data not found! Using synthetic fallback.")
    loader.block_ids = [f"synth_b{i}" for i in range(100)]

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, help="System name (e.g. Cascade, HDF5-Native, PDC)")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--use-mpi-io", action="store_true", help="Force MPI-IO for HDF5")
    parser.add_argument("--run-id", type=str, default=None, help="Unique ID for this run within a job")
    args = parser.parse_args()

    name = args.system
    rid = args.run_id or job_id
    print_rank0(f"\n" + "="*60)
    print_rank0(f"🚀 Standalone Benchmark: {name} (Rank {rank}/{world}, RunID: {rid})")
    print_rank0("="*60)
    
    # 1. Configuration
    config = {}
    
    # For Redis: each rank connects to its OWN node's redis-server (written locally)
    # The read phase will reconnect to target_rank's host.
    my_host = socket.gethostname()
    
    # Collect all hostnames first (needed for redis read phase)
    all_hosts = get_all_hosts(rid)
    
    if name.lower() == "cascade":
        config = {"gpu_capacity_gb": 30.0, "shm_capacity_gb": 140.0, "use_gpu": True}
    elif "redis" in name.lower():
        r_port = int(os.environ.get("REDIS_PORT", 16379))
        # FIX: Each rank writes to its OWN node's Redis (not a shared host)
        # This is necessary because we run redis-server on every node via srun --ntasks-per-node=1
        config = {"host": my_host, "port": r_port}
        # Give some time for redis-server to start
        time.sleep(10)
    elif name.lower() == "lmcache":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/lmcache_store_{rid}"}
    elif name.lower() == "llm-gpu" or name.lower() == "vllm-gpu":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/vllm_store_{rid}"}
    elif name.lower() == "pdc":
        config = {"storage_path": f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/pdc_store_{rid}"}
    elif "hdf5" in name.lower():
        config = {
            "file_path": f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/h5_store_{rid}.h5",
            "use_mpi": args.use_mpi_io
        }
    
    if args.use_mpi_io:
        try:
            from mpi4py import MPI
            if not MPI.Is_initialized():
                pass 
        except ImportError:
            pass

    adapter = get_adapter(name, config)
    if not adapter.initialize():
        print_rank0(f"❌ [{name}] Failed to initialize adapter")
        return
        
    try:
        # Clear phase
        print_rank0(f"[{name}] Clearing storage...")
        adapter.clear()
        file_barrier(f"{name}_cleared", rid)
        
        # 2. Key distribution
        req_keys = loader.block_ids[:args.num_requests]
        my_write_reqs = req_keys[rank::world]
        
        # Write Phase
        print_rank0(f"[{name}] Write Phase: {len(my_write_reqs)} reqs/rank...")
        for rk in my_write_reqs:
            k_data, v_data = loader.load(rk)
            adapter.put(rk, k_data, v_data)
        
        adapter.flush()
        if hasattr(adapter, "sync_metadata"):
            print_rank0(f"[{name}] Synchronizing metadata...")
            adapter.sync_metadata()
        file_barrier(f"{name}_written", rid)
        
        # Propagation Phase
        if name.lower() == "cascade":
            time.sleep(10)
        elif "redis" not in name.lower():
            print_rank0(f"[{name}] Waiting for Lustre/FS stabilization...")
            time.sleep(20)
            store_type = name.lower().split('-')[0]
            force_lustre_sync(REPO_ROOT / "benchmark" / f"{store_type}_store")
            force_lustre_sync(REPO_ROOT / "benchmark" / "tmp")

        # Use a more explicit neighbor fetch pattern
        target_rank = (rank + 1) % world
        my_read_reqs = req_keys[target_rank::world]
        
        if "redis" in name.lower():
            target_host = all_hosts.get(target_rank, "127.0.0.1")
            adapter.close()
            adapter.host = target_host
            adapter.initialize()

        print(f"[Rank {rank}] Read Phase: fetching {len(my_read_reqs)} blocks from rank {target_rank}...", flush=True)
        
        read_latencies = []
        f0 = time.time()
        for rk in my_read_reqs:
            t_req = time.time()
            res = None
            max_attempts = 3 if name.lower() == "cascade" else 30
            for attempt in range(max_attempts): 
                res = adapter.get(rk)
                if res: break
                if "redis" not in name.lower() and name.lower() != "cascade":
                    # Force sync on specific missing key if possible
                    store_type = name.lower().split('-')[0]
                    force_lustre_sync(REPO_ROOT / "benchmark" / f"{store_type}_store" / f"{rk}.kv")
                # FIX: DO NOT call sync_metadata() inside the per-block read loop.
                # sync_metadata() issues MPI_Allgather across ALL nodes → O(N²) overhead.
                # At 16N/32N/64N this caused TTFT to spike to 300-558ms.
                # The single sync_metadata() call after write phase is sufficient.
                time.sleep(0.5)
            
            if res:
                read_latencies.append((time.time() - t_req) * 1000)
            else:
                print(f"[Rank {rank}] {name}: Missed key {rk} from rank {target_rank}")
        
        if read_latencies:
            local_avg_ttft = np.mean(read_latencies)
            local_duration = time.time() - f0
            local_throughput = len(read_latencies) / local_duration
            
            # Aggregate stats across ranks using a temporary file
            stats_dir = REPO_ROOT / "benchmark" / "tmp" / f"stats_{rid}"
            if rank == 0:
                if stats_dir.exists():
                    import shutil
                    shutil.rmtree(stats_dir)
            
            file_barrier(f"{name}_stats_clean", rid)
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            stats_file = stats_dir / f"rank_{rank}.json"
            with open(stats_file, 'w') as f:
                json.dump({"ttft": local_avg_ttft, "thru": local_throughput, "total": len(read_latencies)}, f)
            
            file_barrier(f"{name}_stats_ready", rid)
            
            if rank == 0:
                all_ttfts = []
                all_thrus = []
                total_blocks = 0
                for r in range(world):
                    try:
                        with open(stats_dir / f"rank_{r}.json", 'r') as f:
                            s = json.load(f)
                            all_ttfts.append(s["ttft"])
                            all_thrus.append(s["thru"])
                            total_blocks += s["total"]
                    except: continue
                
                final_avg_ttft = np.mean(all_ttfts)
                final_agg_thru = np.sum(all_thrus)
                
                print(f"\n--- ✨ {name} Benchmark Results ({world} Nodes) ---")
                print(f"Avg TTFT (Per Req): {final_avg_ttft:.2f} ms")
                print(f"Aggregate Throughput: {final_agg_thru:.2f} req/s")
                print(f"Total Read Count: {total_blocks} blocks")
        else:
            print(f"[Rank {rank}] ❌ [{name}] All read requests failed.")
            
    except Exception as e:
        print_rank0(f"❌ [{name}] Crash: {e}")
    finally:
        adapter.close()
    
    file_barrier(f"{name}_done", rid)
    print_rank0(f"\n🏁 Run {rid} complete.")

if __name__ == "__main__":
    run_benchmark()
