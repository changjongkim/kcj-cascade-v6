# benchmark/scripts/v22_multi_turn_inc_bench.py
"""
Benchmark for Multi-Turn Conversation (Incremental Scaling).
Evaluates Incremental Write Latency, Cumulative Read TTFT, and RASE.
"""
import argparse
import time
import os
import numpy as np
import random
from mpi4py import MPI
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.run_benchmark import get_adapter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, required=True)
    parser.add_argument("--num-turns", type=int, default=10, help="Number of conversation turns")
    parser.add_argument("--block-size-mb", type=float, default=160.0)
    parser.add_argument("--sharing-prob", type=float, default=0.7, help="Probability of sharing prefix bytes")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    args = get_args()

    config = {
        "redis_port": args.redis_port,
        "gpu_capacity_gb": 16.0,
        "shm_capacity_gb": 64.0,
        "use_gpu": True
    }
    
    adapter = get_adapter(args.system, config)
    if not adapter.initialize():
        if rank == 0: print(f"Failed to initialize {args.system}")
        return

    block_size = int(args.block_size_mb * 1024 * 1024)
    num_turns = args.num_turns
    
    # Generate data pool
    # Prefix blocks are shared across ranks (simulating a shared system prompt or common topic)
    if rank == 0:
        prefix_data = bytearray(os.urandom(block_size))
    else:
        prefix_data = bytearray(block_size)
    comm.Bcast(prefix_data, root=0)
    
    turn_write_times = []
    turn_read_total_times = []
    logical_bytes_requested = 0
    actual_dedup_hits = 0

    if rank == 0:
        print(f"\n💬 Starting Multi-Turn Scenario: {args.system}")
        print(f"   Nodes: {world_size} | Turns: {num_turns} | Block: {args.block_size_mb} MB")
        print("-" * 60)

    for turn in range(num_turns):
        # 1. Write Turn
        # Turn 0 is always prefix (protected/shared)
        # Subsequent turns are generation blocks (unique to sessions)
        block_id = f"sess_{rank}_turn_{turn}"
        if turn == 0:
            data = prefix_data
            is_prefix = True
        else:
            data = os.urandom(block_size)
            is_prefix = False
            
        mid = len(data) // 2
        key_data = data[:mid]
        val_data = data[mid:]
        
        comm.Barrier()
        t0 = time.time()
        # Note: we use put_prefix for the first block if the adapter supports it
        if turn == 0 and hasattr(adapter, "put_prefix"):
            success = adapter.put_prefix(block_id, key_data, val_data)
        else:
            success = adapter.put(block_id, key_data, val_data)
        t_write = (time.time() - t0) * 1000 # ms
        
        turn_write_times.append(t_write)
        logical_bytes_requested += len(data)

        # 2. Read Turn (Context Reconstruction)
        # Read ALL blocks from turn 0 to current turn
        comm.Barrier()
        t0_read = time.time()
        for t in range(turn + 1):
            target_id = f"sess_{rank}_turn_{t}"
            adapter.get(target_id)
        t_read_total = (time.time() - t0_read) * 1000 # ms
        turn_read_total_times.append(t_read_total)

        avg_write = comm.allreduce(t_write, op=MPI.SUM) / world_size
        avg_read = comm.allreduce(t_read_total, op=MPI.SUM) / world_size
        if rank == 0:
            print(f"Turn {turn:2d}: Write={avg_write:7.2f} ms | ReadTotal={avg_read:7.2f} ms")

    # Final Statistics
    comm.Barrier()
    stats = adapter.get_stats()
    local_dedup_hits = stats.get("dedup_hits", 0)
    total_dedup_hits = comm.allreduce(local_dedup_hits, op=MPI.SUM)
    
    if rank == 0:
        total_logical_gb = (logical_bytes_requested * world_size) / 1024**3
        # RASE (Redundancy-aware Storage Efficiency)
        # If dedup hits are > 0, physical written = Logical - (hits * block_size)
        physical_gb = total_logical_gb - (total_dedup_hits * block_size / 1024**3)
        rase = total_logical_gb / max(physical_gb, 0.001)
        
        print("\n" + "="*60)
        print(f"📊 MULTI-TURN FINAL REPORT: {args.system}")
        print("="*60)
        print(f"  Logical Volume:     {total_logical_gb:.2f} GB")
        print(f"  Physical Volume:    {physical_gb:.2f} GB")
        print(f"  RASE Score:         {rase:.2f}x (Higher is better)")
        print(f"  Avg Write Latency:  {np.mean(turn_write_times):.2f} ms")
        print(f"  Final Read TTFT:    {turn_read_total_times[-1]:.2f} ms (History size {num_turns})")
        print("="*60)

    adapter.close()

if __name__ == "__main__":
    main()
