#!/usr/bin/env python3
"""
Cascade V6 Comprehensive Benchmark
===================================
Tests all V6 features and compares against V5 baseline:
  1. Synthetic throughput (GPU + SHM + Lustre tiers)
  2. Real app data (LLaMA-70B KV cache traces)
  3. Feature ablation: Compression, Aggregated Lustre, Prefetch

Previous V5 Results (for comparison):
  - Synthetic GPU Pinned Write: 23.3 GB/s, Read: 22.2 GB/s
  - Synthetic SHM Write: 18.0 GB/s, Read: 13.5 GB/s
  - Real Data Write: 3.10 GB/s, Read: 2.45 GB/s
  - Dedup Write: 4.33 GB/s, 40 dedup hits
"""

import os
import sys
import json
import time
import struct
import shutil
import numpy as np
from pathlib import Path

REPO_ROOT = Path("/pscratch/sd/s/sgkim/kcj/Cascade-kcj")
CPP_BUILD = REPO_ROOT / "cascade_Code/cpp/build"
sys.path.insert(0, str(CPP_BUILD))
sys.path.insert(0, str(REPO_ROOT))

import cascade_cpp

DATA_DIR = Path("/pscratch/sd/s/sgkim/cascade_kv_cache")
LUSTRE_BASE = REPO_ROOT / "benchmark" / "data"

# ===========================================================================
# Helpers
# ===========================================================================

class RealDataReader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        index_path = data_dir / "global_index.json"
        with open(index_path) as f:
            self.index = json.load(f).get("blocks", {})
    
    def get_block_ids(self):
        return list(self.index.keys())
    
    def read_block(self, block_id: str):
        info = self.index[block_id]
        with open(self.data_dir / info["file"], "rb") as f:
            f.seek(info["offset"])
            header = f.read(16)
            key_size, value_size = struct.unpack("<QQ", header)
            data = f.read(key_size + value_size)
            return np.frombuffer(data, dtype=np.uint8).copy()


def clean_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def fmt(val, unit="GB/s"):
    return f"{val:.2f} {unit}"


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_comparison(metric, v5_val, v6_val, unit="GB/s"):
    change = ((v6_val - v5_val) / v5_val * 100) if v5_val > 0 else 0
    arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
    print(f"  {metric:<30s}  V5: {v5_val:>8.2f}  V6: {v6_val:>8.2f} {unit}  {arrow} {abs(change):.1f}%")


# ===========================================================================
# Part 1: Synthetic Throughput
# ===========================================================================

def run_synthetic_bench():
    print_header("Part 1: Synthetic C++ Backend Throughput")
    print("  (GPU write/read handled by C++ cascade_bench binary)")
    print("  Running SHM-only synthetic test from Python...\n")
    
    SHM_CAP = 2 * 1024 * 1024 * 1024  # 2GB
    BLOCK_SIZE = 1 * 1024 * 1024       # 1MB
    NUM_BLOCKS = 500
    total_bytes = BLOCK_SIZE * NUM_BLOCKS
    total_gb = total_bytes / (1024**3)
    
    results = {}
    
    # --- V5 baseline: per-file Lustre, no compression ---
    for label, use_compression, use_agg in [
        ("V5 Baseline", False, False),
        ("V6 +Compression", True, False),
        ("V6 +AggLustre", False, True),
        ("V6 +Both", True, True),
    ]:
        lustre_path = str(LUSTRE_BASE / f"synth_{label.replace(' ', '_').replace('+', '')}")
        clean_path(lustre_path)
        
        config = cascade_cpp.CascadeConfig()
        config.use_gpu = False
        config.shm_capacity_bytes = SHM_CAP
        config.lustre_path = lustre_path
        config.kv_compression = use_compression
        config.aggregated_lustre = use_agg
        config.prefetch_enabled = False  # measure raw throughput first
        config.semantic_eviction = True
        
        store = cascade_cpp.CascadeStore(config)
        
        # Generate synthetic data
        blocks = []
        for i in range(NUM_BLOCKS):
            data = np.random.randint(0, 255, BLOCK_SIZE, dtype=np.uint8)
            bid = f"synth_block_{i:06d}"
            blocks.append((bid, data))
        
        # Write
        start = time.time()
        for bid, data in blocks:
            store.put(bid, data, False)
        write_dur = time.time() - start
        write_tp = total_gb / write_dur
        
        stats = store.get_stats()
        
        # Read
        out_buf = np.zeros(BLOCK_SIZE * 2, dtype=np.uint8)  # extra space for decompression
        start = time.time()
        for bid, _ in blocks:
            store.get(bid, out_buf)
        read_dur = time.time() - start
        read_tp = total_gb / read_dur
        
        print(f"  [{label}]")
        print(f"    Write: {fmt(write_tp)}  Read: {fmt(read_tp)}")
        print(f"    SHM used: {stats.shm_used/(1024**2):.0f}MB  Evictions: {stats.shm_evictions}")
        if use_compression:
            print(f"    Compression savings: {stats.compression_savings_bytes/(1024**2):.0f}MB")
        if use_agg:
            print(f"    Lustre puts: {stats.lustre_puts}")
        print()
        
        results[label] = {"write": write_tp, "read": read_tp, "stats": stats}
        
        del store
        # cleanup
        shutil.rmtree(lustre_path, ignore_errors=True)
    
    return results


# ===========================================================================
# Part 2: Real Application Data
# ===========================================================================

def run_real_data_bench():
    print_header("Part 2: Real Application Data (LLaMA-70B KV Cache)")
    
    reader = RealDataReader(DATA_DIR)
    block_ids = reader.get_block_ids()
    print(f"  Total blocks in trace: {len(block_ids)}")
    
    NUM_TEST = 100
    test_ids = block_ids[:NUM_TEST]
    
    print(f"  Reading {NUM_TEST} real blocks into memory...")
    blocks = []
    total_bytes = 0
    t0 = time.time()
    for bid in test_ids:
        data = reader.read_block(bid)
        if data is not None:
            blocks.append((bid, data))
            total_bytes += len(data)
    load_time = time.time() - t0
    total_gb = total_bytes / (1024**3)
    print(f"  Loaded {len(blocks)} blocks ({total_gb:.2f} GB) in {load_time:.1f}s\n")
    
    results = {}
    
    for label, use_compression, use_agg, use_prefetch in [
        ("V5 Baseline",       False, False, False),
        ("V6 +Compression",   True,  False, False),
        ("V6 +AggLustre",     False, True,  False),
        ("V6 +Prefetch",      False, False, True),
        ("V6 All Features",   True,  True,  True),
    ]:
        lustre_path = str(LUSTRE_BASE / f"real_{label.replace(' ', '_').replace('+', '')}")
        clean_path(lustre_path)
        
        config = cascade_cpp.CascadeConfig()
        config.use_gpu = False
        config.shm_capacity_bytes = 8 * 1024 * 1024 * 1024  # 8GB
        config.lustre_path = lustre_path
        config.kv_compression = use_compression
        config.aggregated_lustre = use_agg
        config.prefetch_enabled = use_prefetch
        config.prefetch_threads = 2
        config.semantic_eviction = True
        
        store = cascade_cpp.CascadeStore(config)
        
        # --- Write ---
        start = time.time()
        for bid, data in blocks:
            store.put(bid, data, False)
        write_dur = time.time() - start
        write_tp = total_gb / write_dur
        
        stats_w = store.get_stats()
        
        # --- Read ---
        max_block_size = max(len(d) for _, d in blocks) * 2  # extra for decompression
        out_buf = np.zeros(max_block_size, dtype=np.uint8)
        
        start = time.time()
        shm_hits = 0
        lustre_hits = 0
        for bid, _ in blocks:
            found, size = store.get(bid, out_buf)
        read_dur = time.time() - start
        read_tp = total_gb / read_dur
        
        stats_r = store.get_stats()
        shm_hits = stats_r.shm_hits
        lustre_hits = stats_r.lustre_hits
        
        # --- Dedup Test ---
        store.clear()
        clean_path(lustre_path)
        
        # Re-initialize if aggregated (need fresh files)
        if use_agg:
            del store
            store = cascade_cpp.CascadeStore(config)
        
        prefixes = blocks[:10]
        unique = blocks[10:30]
        
        all_reqs = []
        for user in range(5):
            for bid, data in prefixes:
                all_reqs.append((bid, data, True))
            for bid, data in unique:
                all_reqs.append((f"user{user}_u_{bid}", data, False))
        
        total_dedup_bytes = sum(len(d) for _, d, _ in all_reqs)
        total_dedup_gb = total_dedup_bytes / (1024**3)
        
        start = time.time()
        for bid, data, is_pref in all_reqs:
            store.put(bid, data, is_pref)
        dedup_dur = time.time() - start
        dedup_tp = total_dedup_gb / dedup_dur
        
        stats_d = store.get_stats()
        
        print(f"  [{label}]")
        print(f"    Write:  {fmt(write_tp)}  (SHM evictions: {stats_w.shm_evictions})")
        print(f"    Read:   {fmt(read_tp)}  (SHM hits: {shm_hits}, Lustre hits: {lustre_hits})")
        print(f"    Dedup:  {fmt(dedup_tp)}  (dedup hits: {stats_d.dedup_hits})")
        if use_compression:
            print(f"    Compression savings: {stats_w.compression_savings_bytes/(1024**2):.0f}MB")
        if use_prefetch:
            print(f"    Prefetch completed: {stats_r.prefetch_completed}")
        print()
        
        results[label] = {
            "write": write_tp,
            "read": read_tp,
            "dedup": dedup_tp,
            "dedup_hits": stats_d.dedup_hits,
            "compression_savings": stats_w.compression_savings_bytes,
            "evictions": stats_w.shm_evictions,
        }
        
        del store
        shutil.rmtree(lustre_path, ignore_errors=True)
    
    return results


# ===========================================================================
# Part 3: Summary Comparison
# ===========================================================================

def print_summary(synth_results, real_results):
    print_header("Part 3: V5 vs V6 Comparison Summary")
    
    print("\n  ┌─── Synthetic (500 × 1MB blocks, SHM 2GB) ────────────────────────┐")
    if "V5 Baseline" in synth_results and "V6 +Both" in synth_results:
        v5 = synth_results["V5 Baseline"]
        v6 = synth_results["V6 +Both"]
        print_comparison("Write Throughput", v5["write"], v6["write"])
        print_comparison("Read Throughput", v5["read"], v6["read"])
    
    print("\n  ┌─── Real App Data (100 × LLaMA-70B blocks, SHM 8GB) ──────────────┐")
    if "V5 Baseline" in real_results and "V6 All Features" in real_results:
        v5 = real_results["V5 Baseline"]
        v6 = real_results["V6 All Features"]
        print_comparison("Write Throughput", v5["write"], v6["write"])
        print_comparison("Read Throughput", v5["read"], v6["read"])
        print_comparison("Dedup Throughput", v5["dedup"], v6["dedup"])
        print_comparison("Dedup Hits", v5["dedup_hits"], v6["dedup_hits"], "blocks")
        print_comparison("Compression Savings", 0, v6["compression_savings"]/(1024**2), "MB")
        print_comparison("SHM Evictions", v5["evictions"], v6["evictions"], "blocks")
    
    print("\n  ┌─── Feature Breakdown ─────────────────────────────────────────────┐")
    for label in ["V5 Baseline", "V6 +Compression", "V6 +AggLustre", "V6 +Prefetch", "V6 All Features"]:
        if label in real_results:
            r = real_results[label]
            print(f"    {label:<22s}  W: {r['write']:.2f}  R: {r['read']:.2f}  D: {r['dedup']:.2f} GB/s")
    
    print()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + "  Cascade V6 Comprehensive Benchmark".center(68) + "║")
    print("║" + "  Async Prefetch + INT8 Compression + Aggregated Lustre I/O".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    synth_results = run_synthetic_bench()
    real_results = run_real_data_bench()
    print_summary(synth_results, real_results)
    
    print("=" * 70)
    print("✅ V6 Benchmark Complete!")
    print("=" * 70)
