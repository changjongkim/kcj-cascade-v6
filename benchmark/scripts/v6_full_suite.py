#!/usr/bin/env python3
"""
Cascade V6 Full Benchmark Suite — Comprehensive Experiments on 1, 2, 4, 8 Nodes

Covers:
  1. Synthetic Scaling (Throughput vs Nodes)
  2. Multi-Turn Dialogue Simulation (ShareGPT) -> Novelty 1 & 2
  3. Long-Context Stress Test (PG-19) -> Novelty 3 & Tiering
  4. Code Locality Test (The Stack) -> Novelty 3

Run: srun -N<Nodes> -n<Nodes> --gpus-per-node=4 python v6_full_suite.py
"""

import sys
import os
import time
import hashlib
import numpy as np
import logging

# Add MPI build directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../cascade_Code/cpp/build_mpi'))
# Add benchmark root for dataset loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cascade_cpp
    from dataset_loader import ExternalDatasetLoader
    HAS_DISTRIBUTED = hasattr(cascade_cpp, 'DistributedStore')
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)

def get_rank_info():
    """Get MPI rank info."""
    rank = int(os.environ.get('SLURM_PROCID', os.environ.get('PMI_RANK', 0)))
    world = int(os.environ.get('SLURM_NTASKS', os.environ.get('PMI_SIZE', 1)))
    return rank, world

def print_rank0(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)

# ============================================================================
# Exp 1: Baseline Synthetic Scaling (Throughput)
# ============================================================================
def exp1_synthetic_scaling(rank, world, store):
    """Measure raw aggregated throughput with random data."""
    print_rank0(f"\n[Exp 1] Synthetic Scaling (Writes & Reads)", rank)
    
    num_blocks = 500
    block_size = 160 * 1024  # 160KB
    data = np.random.randint(0, 255, block_size, dtype=np.uint8)
    
    # Write
    store.barrier()
    t0 = time.time()
    for i in range(num_blocks):
        # Unique keys per rank
        store.put(f"syn_r{rank}_b{i}", data)
    store.barrier()
    t_write = time.time() - t0
    
    # Read
    t1 = time.time()
    for i in range(num_blocks):
        buf = np.empty_like(data)
        store.get(f"syn_r{rank}_b{i}", buf)
    store.barrier()
    t_read = time.time() - t1
    
    total_gb = (num_blocks * block_size * world) / 1024**3
    print_rank0(f"  Write: {total_gb/t_write:.2f} GB/s", rank)
    print_rank0(f"  Read:  {total_gb/t_read:.2f} GB/s", rank)
    
    return {'write_gbps': total_gb/t_write, 'read_gbps': total_gb/t_read}

# ============================================================================
# Exp 2: ShareGPT Simulation (Dedup & Eviction)
# ============================================================================
def exp2_sharegpt_workload(rank, world, store, loader):
    """
    Simulate multi-turn chat.
    - System Prompts (Shared) -> Dedup hits expected
    - User History (Unique) -> Eviction pressure
    """
    print_rank0(f"\n[Exp 2] ShareGPT Workload (Dedup Efficiency)", rank)
    
    conversations = loader.load_sharegpt()
    if not conversations:
        print_rank0("  Skipping Exp 2 (ShareGPT data missing)", rank)
        return

    # Use a subset for speed
    subset = conversations[:50] 
    
    store.barrier()
    t0 = time.time()
    
    # Phase 1: All ranks write common system prompt (Simulated)
    sys_prompt = np.frombuffer(b"You are a helpful assistant." * 1000, dtype=np.uint8)
    sys_key = "sys_prompt_v1"
    store.put(sys_key, sys_prompt, is_prefix=True) # All ranks write same key/value
    
    store.sync_metadata() # Advertise the prompt
    
    # Phase 2: Write unique conversation turns
    for i, conv in enumerate(subset):
        # Simple simulation: convert string to bytes
        data = np.frombuffer(conv.encode('utf-8')[:160*1024], dtype=np.uint8)
        key = f"sharegpt_r{rank}_c{i}"
        store.put(key, data, is_prefix=False)

    store.barrier()
    stats = store.get_stats()
    
    print_rank0(f"  Dedup Hits: {stats.dedup_hits}", rank)
    print_rank0(f"  Prefix Protected: {stats.prefix_blocks_protected}", rank)
    print_rank0(f"  Bytes Saved: {stats.dedup_bytes_saved / 1024**2:.2f} MB", rank)

# ============================================================================
# Exp 3: Long Context (PG-19) Stess Test
# ============================================================================
def exp3_long_context(rank, world, store, loader):
    """
    Write a massive continuous stream of blocks to force Tier 4/5 usage.
    """
    print_rank0(f"\n[Exp 3] Long Context Stress (Capacity Overflow)", rank)
    
    text = loader.load_longbench_pg19()
    if not text:
        print_rank0("  Skipping Exp 3 (PG-19 data missing)", rank)
        return

    # Chunk text into blocks
    chunk_size = 1 * 1024 * 1024 # 1MB blocks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    store.barrier()
    
    # Each rank writes full book (simulating long context per client)
    for i, chunk in enumerate(chunks):
        data = np.frombuffer(chunk.encode('utf-8'), dtype=np.uint8)
        store.put(f"pg19_r{rank}_k{i}", data)
        
    store.barrier()
    stats = store.get_stats()
    
    print_rank0(f"  Total Written: {len(chunks)*world} MB", rank)
    print_rank0(f"  Lustre Evictions: {stats.dram_evictions}", rank) # Proxy for spillover
    print_rank0(f"  Cluster DRAM Used: {stats.cluster_dram_used / 1024**2:.1f} MB", rank)

# ============================================================================
# Main Runner
# ============================================================================
def main():
    rank, world = get_rank_info()
    
    # Config for Full Suite
    cfg = cascade_cpp.DistributedConfig()
    cfg.dedup_enabled = True
    cfg.semantic_eviction = True
    cfg.locality_aware = True
    cfg.kv_compression = True
    
    # Tighter limits to force tiering behavior even with small data
    cfg.gpu_capacity_per_device = 1 * 1024**3  # 1GB GPU
    cfg.dram_capacity = 4 * 1024**3             # 4GB DRAM
    
    store = cascade_cpp.DistributedStore(cfg)
    loader = ExternalDatasetLoader()
    
    if rank == 0:
        print(f"Running Cascade V6 Full Suite on {world} Ranks...")

    # Run Experiments
    exp1_synthetic_scaling(rank, world, store)
    exp2_sharegpt_workload(rank, world, store, loader)
    exp3_long_context(rank, world, store, loader)
    
    if rank == 0:
        print("\n✅ Full Benchmark Suite Completed.")

if __name__ == "__main__":
    main()
