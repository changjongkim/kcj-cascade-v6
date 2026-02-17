import os
import sys
import time
import numpy as np
import argparse
import json
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
        
        if not index_path.exists():
            print(f"Warning: Index file not found at {index_path}. Using fallback synthetic data.")
            self.use_synthetic = True
            return
            
        with open(index_path, 'r') as f:
            data = json.load(f)
            self.all_blocks = data['blocks']
            self.block_ids = list(self.all_blocks.keys())
        self.use_synthetic = False
            
    def read_block(self, block_id):
        if self.use_synthetic:
            # Fallback: Generate 160MB block
            return np.random.randint(0, 255, 160 * 1024 * 1024, dtype=np.uint8)
            
        loc = self.all_blocks[block_id]
        path = self.base_dir / loc['file']
        with open(path, 'rb') as f:
            f.seek(loc['offset'])
            return np.frombuffer(f.read(loc['size']), dtype=np.uint8)

def run_rdma_microbench():
    print_rank0(f"\n{'='*80}")
    print_rank0(f" CASCADE V6 READ-ONLY BANDWIDTH BENCHMARK | Nodes: {world}")
    print_rank0(f"{'='*80}")

    # 1. Initialize Cascade (Disable Compression/Dedup to measure pure RDMA?)
    # Actually we want to measure WITH default features, but avoid 100% hit.
    cfg = cascade_cpp.DistributedConfig()
    cfg.gpu_capacity_per_device = 38 * 1024**3
    cfg.dram_capacity = 128 * 1024**3
    cfg.num_gpus_per_node = 4
    cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/rdma_rand_{world}"
    store = cascade_cpp.DistributedStore(cfg)
    
    # 2. Generate Unique Random Data per Rank
    # 160MB is a typical block size for LLaMA-3-70B
    block_size = 160 * 1024 * 1024
    
    # Critical: Seed with rank to ensure unique content across ranks
    np.random.seed(rank + int(time.time()) + 12345)
    data_block = np.random.randint(0, 255, block_size, dtype=np.uint8)
    key_name = f"blk_{rank}"
    
    if rank == 0:
        print(f" [Data] Generated Unique Random Block {block_size/1024/1024:.2f} MB")

    # 3. Put Data (Populate Local Store)
    store.put(key_name, data_block)
    store.barrier()
    
    # 4. Shifted Read Pattern (Force Remote Read)
    # Rank i reads from Rank (i+1)%world
    target_rank = (rank + 1) % world
    target_key = f"blk_{target_rank}"
    
    out_buf = np.empty(block_size * 2, dtype=np.uint8) # Buffer with safety margin

    # Warmup
    for _ in range(5):
        store.get(target_key, out_buf)
    store.barrier()

    # Measure
    iters = 20
    t_start = time.time()
    total_bytes = 0
    
    for _ in range(iters):
        ret = store.get(target_key, out_buf)
        if isinstance(ret, tuple):
             total_bytes += ret[1] # size read
        else:
             total_bytes += block_size 

    store.barrier()
    t_end = time.time()
    
    duration = t_end - t_start
    # Calculate per-rank throughput
    my_bw_gbps = (total_bytes / 1024.0**3) / duration
    
    # Print per-rank stats
    print(f" [Rank {rank}] Read from Rank {target_rank} | BW: {my_bw_gbps:.2f} GB/s")
    store.barrier()
    
    # Summary (Estimate based on Rank 0)
    if rank == 0:
        est_aggr_bw = my_bw_gbps * world
        print(f"\n [Summary] Estimated Aggregate RDMA BW: {est_aggr_bw:.2f} GB/s")
        print(f" [Summary] Per-node RDMA BW:          {my_bw_gbps:.2f} GB/s")

    store.barrier()
    
    # Cleanup
    if rank == 0:
        import shutil
        if os.path.exists(cfg.lustre_path): 
            try: shutil.rmtree(cfg.lustre_path)
            except: pass
    print_rank0(f"{'='*80}\n")

if __name__ == "__main__":
    run_rdma_microbench()
