#!/usr/bin/env python3
"""
TTFT (Time to First Token) & TBT (Time Between Tokens) Benchmark
=================================================================
Simulates LLM inference-level metrics using Cascade and competitor
KV cache storage systems.

TTFT: Measures the time to load all prefix + context KV cache blocks
      from storage into GPU memory before the first token can be generated.
      → Maps to the "Prefill" phase of LLM inference.

TBT:  Measures the per-token overhead during autoregressive decode,
      including KV cache store (put) and potential eviction/fetch (get).
      → Maps to the "Decode" phase of LLM inference.

Usage:
  # TTFT: Measure prefill latency for 16 prefix blocks (Cold Start)
  srun python3 benchmark/scripts/v6_ttft_tbt_benchmark.py \\
      --metric ttft --model qwen-2.5-72b \\
      --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \\
      --prefix-blocks 16 --cold

  # TBT: Measure per-token decode overhead with 100 steps
  srun python3 benchmark/scripts/v6_ttft_tbt_benchmark.py \\
      --metric tbt --model qwen-2.5-72b \\
      --systems Cascade,HDF5,vLLM-GPU,PDC,LMCache \\
      --decode-steps 100 --concurrent-requests 4
"""
import os
import sys
import time
import json
import numpy as np
import argparse
from pathlib import Path

# ============================================================
# MPI Configuration (SLURM env vars)
# ============================================================
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
# Storage Adapters (same as v6_contention_scaling_all.py)
# ============================================================

class BaseStore:
    def put(self, key, data): pass
    def get(self, key, out): pass
    def barrier(self): pass
    def cleanup(self): pass

GLOBAL_STORE = None

def init_global_store(tag="ttft"):
    global GLOBAL_STORE
    if GLOBAL_STORE is None:
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 160 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_ttft_{tag}_{job_id}"
        GLOBAL_STORE = cascade_cpp.DistributedStore(cfg)
    return GLOBAL_STORE

def mpi_barrier():
    if GLOBAL_STORE:
        GLOBAL_STORE.barrier()

class CascadeAdapter(BaseStore):
    def __init__(self, tag):
        self.store = GLOBAL_STORE
        self.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_ttft_{tag}"

    def put(self, key, data):
        self.store.put(str(key), data)

    def get(self, key, out):
        found, size = self.store.get(str(key), out)
        return found

    def barrier(self):
        self.store.barrier()

    def cleanup(self):
        self.barrier()
        if rank == 0:
            import shutil
            if os.path.exists(self.lustre_path):
                try:
                    shutil.rmtree(self.lustre_path)
                except:
                    pass

class HDF5Adapter(BaseStore):
    def __init__(self, tag):
        import h5py
        self.h5py = h5py
        self.shared_path = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_ttft_{tag}_{job_id}.h5")
        self.local_path = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_ttft_{tag}_{job_id}_r{rank}.h5")
        self.file_write = None
        self.file_read_shared = None
        self.file_read_local = None
        if rank == 0 and self.shared_path.exists(): self.shared_path.unlink()
        if self.local_path.exists(): self.local_path.unlink()
        mpi_barrier()

    def put(self, key, data):
        is_shared = "rank" not in str(key) and "node" not in str(key)
        if is_shared:
            if rank == 0:
                if not self.file_write: self.file_write = self.h5py.File(self.shared_path, 'a')
                if str(key) not in self.file_write:
                    self.file_write.create_dataset(str(key), data=data)
                    self.file_write.flush()
        else:
            if not self.file_write: self.file_write = self.h5py.File(self.local_path, 'a')
            if str(key) not in self.file_write:
                self.file_write.create_dataset(str(key), data=data)
                self.file_write.flush()

    def flush(self):
        if self.file_write:
            self.file_write.flush()
            self.file_write.close()
            self.file_write = None
        mpi_barrier()
        if rank == 0:
            time.sleep(2)
        mpi_barrier()

    def open_for_read(self):
        if self.file_write:
            self.file_write.close()
            self.file_write = None
        mpi_barrier()
        time.sleep(3)
        if self.shared_path.exists():
            try:
                self.file_read_shared = self.h5py.File(self.shared_path, 'r')
            except Exception as e:
                print(f"[Rank {rank}] Error opening shared HDF5: {e}")
                time.sleep(2)
                try: self.file_read_shared = self.h5py.File(self.shared_path, 'r')
                except: pass
        if self.local_path.exists():
            self.file_read_local = self.h5py.File(self.local_path, 'r')

    def get(self, key, out):
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
    def __init__(self, name, tag):
        import shutil
        self.shared_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_ttft_{tag}_{job_id}")
        self.local_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_ttft_{tag}_{job_id}_r{rank}")
        if rank == 0:
            if self.shared_dir.exists(): shutil.rmtree(self.shared_dir)
            self.shared_dir.mkdir(parents=True, exist_ok=True)
        if self.local_dir.exists(): shutil.rmtree(self.local_dir)
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
# Model Config
# ============================================================

def get_model_config(model_name):
    """Returns block size in bytes.  1 block = 1024 tokens of KV cache."""
    configs = {
        "llama-3-70b":  (80, 8, 128, 1024),
        "qwen-2.5-72b": (80, 8, 128, 1024),
        "qwen-2.5-32b": (64, 8, 128, 1024),
        "qwen-2.5-7b":  (28, 4, 128, 1024),
    }
    L, H, D, T = configs.get(model_name, configs["llama-3-70b"])
    return 2 * 2 * L * H * D * T   # fp16 * (K+V) * layers * heads * dim * tokens

def tokens_per_block():
    """Each block stores 1024 tokens."""
    return 1024

# ============================================================
# Adapter Factory
# ============================================================

def make_adapter(name, tag):
    if name == "Cascade":    return CascadeAdapter(tag)
    elif name == "HDF5":     return HDF5Adapter(tag)
    elif name == "vLLM-GPU": return PosixAdapter("vllm", tag)
    elif name == "PDC":      return PosixAdapter("pdc", tag)
    elif name == "LMCache":  return PosixAdapter("lmcache", tag)
    return None

# ============================================================
# TTFT Benchmark (Prefill Phase Simulation)
# ============================================================

def run_ttft_benchmark(args, systems, block_size):
    """
    Simulate TTFT: Time to load N prefix blocks from storage → GPU.

    In a real LLM serving scenario:
      1. User request arrives with a system prompt + context
      2. Engine checks KV cache for existing prefix blocks
      3. TTFT = time to load ALL required blocks before first token generation

    We measure:
      - Total TTFT (sum of all block loads)
      - Per-block latency distribution (p50, p95, p99)
      - First-block latency (time to very first data availability)
    """
    num_prefix = args.prefix_blocks
    total_context_tokens = num_prefix * tokens_per_block()
    total_data_mb = (num_prefix * block_size) / 1024**2
    total_data_gb = total_data_mb / 1024

    print_rank0(f"\n{'='*90}")
    print_rank0(f"  TTFT BENCHMARK (Prefill Phase Simulation)")
    print_rank0(f"  Model: {args.model.upper()} | Block Size: {block_size/1024**2:.0f} MB")
    print_rank0(f"  Prefix Blocks: {num_prefix} ({total_context_tokens:,} tokens)")
    print_rank0(f"  Total Prefill Data: {total_data_mb:.1f} MB per rank | {total_data_gb*world:.2f} GB cluster-wide")
    print_rank0(f"  Mode: {'COLD (Lustre → GPU)' if args.cold else 'HOT (Cache → GPU)'}")
    print_rank0(f"{'='*90}")

    # Generate prefix data (shared across all ranks = system prompt scenario)
    prefix_keys = [f"prefix_b{i}" for i in range(num_prefix)]
    data_cache = {k: generate_block(abs(hash(k)) % (2**32), block_size) for k in prefix_keys}

    # Header
    print_rank0(f"{'System':12} | {'TTFT (ms)':>12} | {'1st Block (ms)':>15} | "
                f"{'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10} | "
                f"{'Prefill BW':>12}")
    print_rank0("-" * 100)

    for name in systems:
        tag = f"ttft_{name}_{num_prefix}"
        adapter = make_adapter(name, tag)
        if not adapter:
            continue

        # === Phase 1: Populate Storage (Rank 0 writes prefix blocks) ===
        if rank == 0:
            for k in prefix_keys:
                adapter.put(k, data_cache[k])

        if hasattr(adapter, 'flush'):
            adapter.flush()
        elif hasattr(adapter, 'store') and hasattr(adapter.store, 'flush'):
            adapter.store.flush()

        adapter.barrier()

        # === Phase 2: Cold Start (optional) ===
        if args.cold:
            if name == "Cascade":
                adapter.store.clear()
            adapter.barrier()

        if hasattr(adapter, 'open_for_read'):
            adapter.open_for_read()

        # === Phase 3: TTFT Measurement ===
        # Each rank loads ALL prefix blocks sequentially (simulates prefill)
        block_latencies = []
        adapter.barrier()

        ttft_start = time.time()
        for k in prefix_keys:
            t0 = time.time()
            buf = np.empty(block_size, dtype=np.uint8)
            adapter.get(k, buf)
            lat_ms = (time.time() - t0) * 1000
            block_latencies.append(lat_ms)

        adapter.barrier()
        ttft_total_ms = (time.time() - ttft_start) * 1000

        # === Statistics ===
        lats = np.array(block_latencies)
        first_block_ms = lats[0] if len(lats) > 0 else 0
        p50 = np.percentile(lats, 50) if len(lats) > 0 else 0
        p95 = np.percentile(lats, 95) if len(lats) > 0 else 0
        p99 = np.percentile(lats, 99) if len(lats) > 0 else 0
        prefill_bw = (total_data_gb * world) / (ttft_total_ms / 1000) if ttft_total_ms > 0 else 0

        print_rank0(f"{name:12} | {ttft_total_ms:12.2f} | {first_block_ms:15.2f} | "
                    f"{p50:10.2f} | {p95:10.2f} | {p99:10.2f} | "
                    f"{prefill_bw:10.2f} GB/s")

        adapter.cleanup()

    print_rank0(f"{'='*90}\n")

# ============================================================
# TBT Benchmark (Decode Phase Simulation)
# ============================================================

def run_tbt_benchmark(args, systems, block_size):
    """
    Simulate TBT: Per-token overhead during autoregressive decode.

    In a real LLM serving scenario:
      1. After prefill, the engine generates tokens one by one
      2. Each token generates new KV entries (periodically flushed as blocks)
      3. Attention may need to fetch previously evicted KV blocks
      4. TBT = storage overhead per decode step

    We model each "decode step" as:
      - 1× store.get()  → Read a context block (attention over past KV)
      - 1× store.put()  → Write a new KV block (new token KV accumulation)

    Variables:
      - decode_steps: Number of decode iterations to simulate
      - concurrent_requests: Simulated as each rank running its own decode stream
    """
    num_ctx = args.prefix_blocks          # context blocks already loaded
    decode_steps = args.decode_steps
    concurrent = args.concurrent_requests  # each rank simulates this many streams

    print_rank0(f"\n{'='*90}")
    print_rank0(f"  TBT BENCHMARK (Decode Phase Simulation)")
    print_rank0(f"  Model: {args.model.upper()} | Block Size: {block_size/1024**2:.0f} MB")
    print_rank0(f"  Context Blocks: {num_ctx} | Decode Steps: {decode_steps}")
    print_rank0(f"  Concurrent Streams per Rank: {concurrent} | Total Streams: {concurrent * world}")
    print_rank0(f"  Mode: {'COLD' if args.cold else 'HOT'}")
    print_rank0(f"{'='*90}")

    # Pre-generate context blocks (shared prefix) + per-rank decode blocks
    ctx_keys = [f"ctx_b{i}" for i in range(num_ctx)]
    data_cache = {k: generate_block(abs(hash(k)) % (2**32), block_size) for k in ctx_keys}

    # Pre-generate decode write data (each rank writes unique blocks)
    decode_write_keys = []
    decode_write_data = {}
    for s in range(concurrent):
        for step in range(decode_steps):
            k = f"dec_r{rank}_s{s}_t{step}"
            decode_write_keys.append((s, step, k))
            decode_write_data[k] = generate_block(abs(hash(k)) % (2**32), block_size)

    # Header
    print_rank0(f"{'System':12} | {'Avg TBT (ms)':>14} | {'p50 (ms)':>10} | "
                f"{'p95 (ms)':>10} | {'p99 (ms)':>10} | "
                f"{'Read (ms)':>10} | {'Write (ms)':>11} | "
                f"{'Decode BW':>12}")
    print_rank0("-" * 115)

    for name in systems:
        tag = f"tbt_{name}_{num_ctx}"
        adapter = make_adapter(name, tag)
        if not adapter:
            continue

        # === Phase 1: Populate context blocks ===
        if rank == 0:
            for k in ctx_keys:
                adapter.put(k, data_cache[k])

        if hasattr(adapter, 'flush'):
            adapter.flush()
        elif hasattr(adapter, 'store') and hasattr(adapter.store, 'flush'):
            adapter.store.flush()
        adapter.barrier()

        if args.cold:
            if name == "Cascade":
                adapter.store.clear()
            adapter.barrier()

        if hasattr(adapter, 'open_for_read'):
            adapter.open_for_read()

        # === Phase 2: TBT Measurement ===
        # Simulate concurrent decode streams sequentially per rank
        # (In production, these would be threaded, but storage is the bottleneck)
        step_latencies = []
        read_latencies = []
        write_latencies = []

        adapter.barrier()
        tbt_start = time.time()

        rng = np.random.RandomState(rank * 1000 + 42)

        for s in range(concurrent):
            for step in range(decode_steps):
                step_t0 = time.time()

                # --- Read: Attention over a random past context block ---
                read_key = ctx_keys[rng.randint(0, len(ctx_keys))]
                buf = np.empty(block_size, dtype=np.uint8)
                r_t0 = time.time()
                adapter.get(read_key, buf)
                read_latencies.append((time.time() - r_t0) * 1000)

                # --- Write: Store new KV block for this decode step ---
                write_key = f"dec_r{rank}_s{s}_t{step}"
                w_t0 = time.time()
                adapter.put(write_key, decode_write_data[write_key])
                write_latencies.append((time.time() - w_t0) * 1000)

                step_latencies.append((time.time() - step_t0) * 1000)

        adapter.barrier()
        tbt_total_s = time.time() - tbt_start

        # === Statistics ===
        steps = np.array(step_latencies)
        reads = np.array(read_latencies)
        writes = np.array(write_latencies)

        avg_tbt = np.mean(steps) if len(steps) > 0 else 0
        p50 = np.percentile(steps, 50) if len(steps) > 0 else 0
        p95 = np.percentile(steps, 95) if len(steps) > 0 else 0
        p99 = np.percentile(steps, 99) if len(steps) > 0 else 0
        avg_read = np.mean(reads) if len(reads) > 0 else 0
        avg_write = np.mean(writes) if len(writes) > 0 else 0

        total_ops =  concurrent * decode_steps * 2  # read + write per step
        total_data_gb = (total_ops * block_size * world) / 1024**3
        decode_bw = total_data_gb / tbt_total_s if tbt_total_s > 0 else 0

        print_rank0(f"{name:12} | {avg_tbt:14.2f} | {p50:10.2f} | "
                    f"{p95:10.2f} | {p99:10.2f} | "
                    f"{avg_read:10.2f} | {avg_write:11.2f} | "
                    f"{decode_bw:10.2f} GB/s")

        adapter.cleanup()

    print_rank0(f"{'='*90}\n")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TTFT / TBT Benchmark for KV Cache Storage Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # TTFT with 16 prefix blocks (Cold Start)
  srun python3 %(prog)s --metric ttft --prefix-blocks 16 --cold

  # TBT with 100 decode steps, 4 concurrent streams
  srun python3 %(prog)s --metric tbt --decode-steps 100 --concurrent-requests 4

  # Both metrics
  srun python3 %(prog)s --metric both --prefix-blocks 16 --decode-steps 50
        """)

    parser.add_argument("--metric", choices=["ttft", "tbt", "both"], required=True,
                        help="Which metric to benchmark")
    parser.add_argument("--model", default="qwen-2.5-72b",
                        choices=["llama-3-70b", "qwen-2.5-72b", "qwen-2.5-32b", "qwen-2.5-7b"])
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache",
                        help="Comma-separated list of storage systems to compare")
    parser.add_argument("--cold", action="store_true",
                        help="Force cold start (clear caches before read)")

    # TTFT-specific
    parser.add_argument("--prefix-blocks", type=int, default=16,
                        help="Number of prefix blocks to load (1 block = 1024 tokens)")

    # TBT-specific
    parser.add_argument("--decode-steps", type=int, default=50,
                        help="Number of decode steps to simulate per stream")
    parser.add_argument("--concurrent-requests", type=int, default=1,
                        help="Number of concurrent decode streams per rank")

    args = parser.parse_args()

    # Initialize Global Store
    store = init_global_store(f"{args.metric}_{args.model}")
    store.barrier()

    systems = args.systems.split(",")
    block_size = get_model_config(args.model)

    print_rank0(f"\n{'#'*90}")
    print_rank0(f"  TTFT / TBT INFERENCE-LEVEL BENCHMARK")
    print_rank0(f"  Model: {args.model} | Block: {block_size/1024**2:.0f} MB | Nodes: {world}")
    print_rank0(f"  Systems: {', '.join(systems)}")
    print_rank0(f"{'#'*90}")

    if args.metric in ("ttft", "both"):
        run_ttft_benchmark(args, systems, block_size)

    if args.metric in ("tbt", "both"):
        run_tbt_benchmark(args, systems, block_size)

    print_rank0("All TTFT/TBT benchmarks completed.")

if __name__ == "__main__":
    main()
