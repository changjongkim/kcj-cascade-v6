#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import argparse

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

import cascade_cpp

def print_rank0(msg):
    if rank == 0:
        print(msg, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=world)
    args = parser.parse_args()

    # Cascade Distributed Store Initialization
    cfg = cascade_cpp.DistributedConfig()
    cfg.gpu_capacity_per_device = 38 * 1024**3
    cfg.dram_capacity = 120 * 1024**3
    cfg.num_gpus_per_node = 4
    cfg.dedup_enabled = False # To measure pure RDMA
    cfg.kv_compression = False
    
    store = cascade_cpp.DistributedStore(cfg)
    store.barrier()

    # Message sizes to sweep (from 1MB up to 512MB)
    msg_sizes_mb = [1, 4, 16, 64, 128, 256, 512]
    iters = 10

    print_rank0("="*90)
    print_rank0(f" CASCADE V6 ADVANCED RDMA BENCHMARK | Nodes: {world}")
    print_rank0("="*90)
    print_rank0(f"{'Msg Size (MB)':<15} | {'Aggr. BW (GB/s)':<20} | {'Avg Latency (ms)':<20} | {'Efficiency'}")
    print_rank0("-" * 90)

    for mb in msg_sizes_mb:
        size = mb * 1024 * 1024
        
        # 1. Prepare unique data on each rank
        # Data is unique to bypass de-duplication
        local_data = np.random.randint(0, 255, size, dtype=np.uint8)
        key = f"rdma_n{rank}_s{mb}"
        store.put(key, local_data, size)
        
        store.barrier()
        
        # 2. Measure Remote Read (RDMA Get)
        # Rank i reads from Rank (i+1)%world
        target_rank = (rank + 1) % world
        target_key = f"rdma_n{target_rank}_s{mb}"
        
        out_buf = np.empty(size, dtype=np.uint8)
        
        latencies = []
        # Warmup
        for _ in range(2):
            store.get(target_key, out_buf)
            
        store.barrier()
        t_start = time.time()
        
        for _ in range(iters):
            t0 = time.time()
            store.get(target_key, out_buf)
            latencies.append((time.time() - t0) * 1000)
            
        store.barrier()
        t_total = time.time() - t_start
        
        # Aggregate stats
        local_avg_lat = np.mean(latencies)
        # Total data read over the cluster in one iteration
        total_data_gb = (size * world) / 1024**3
        aggr_bw = (total_data_gb * iters) / t_total
        
        # Gather stats at Rank 0
        all_bw = comm.gather(aggr_bw, root=0) # Note: bandwidth is already aggregate-calculated
        all_lats = comm.gather(local_avg_lat, root=0)
        
        if rank == 0:
            final_bw = np.mean(all_bw)
            final_lat = np.mean(all_lats)
            # Max theoretical BW for N nodes on Slingshot-11 is Node * 25 GB/s (Single NIC)
            # or Node * 100 GB/s (4 NICs). We use 25GB/s per node as baseline for 1 rank/node.
            max_p2p = world * 12.5 # Estimated realistic P2P limit (half-duplex)
            efficiency = (final_bw / max_p2p) * 100 if world > 1 else (final_bw/25.0)*100
            
            print_rank0(f"{mb:<15} | {final_bw:<20.2f} | {final_lat:<20.2f} | {efficiency:.1f}%")

    print_rank0("="*90)

if __name__ == "__main__":
    main()
