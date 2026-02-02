#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_tiered_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_tiered_%j.err
#SBATCH -J real_tiered

###############################################################################
# REAL TIERED BENCHMARK - Using actual mmap for SHM
# Tests: SHM (mmap on /dev/shm) vs Lustre with proper cold read
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade

module load python/3.11
module load cudatoolkit
module load cray-mpich

cd $PROJECT_DIR
mkdir -p benchmark/results benchmark/logs

JOB_ID=$SLURM_JOB_ID
NPROCS=$SLURM_NTASKS

echo "============================================"
echo "REAL TIERED BENCHMARK"
echo "Job ID: $JOB_ID, Nodes: $SLURM_NNODES, Ranks: $NPROCS"
echo "============================================"

###############################################################################
# Python Benchmark with Real mmap SHM
###############################################################################
srun python3 << 'PYTHON_SCRIPT'
import os
import sys
import time
import json
import mmap
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
import shutil
import struct
import subprocess

RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '/pscratch/sd/s/sgkim/Skim-cascade')

BLOCK_SIZE = 10 * 1024 * 1024  # 10MB per block

# Scenarios: varying SHM capacity to force overflow
SCENARIOS = [
    {"name": "all_shm", "num_blocks": 30, "shm_capacity_mb": 500},       # 300MB data, 500MB SHM = 0% overflow
    {"name": "overflow_50pct", "num_blocks": 50, "shm_capacity_mb": 250}, # 500MB data, 250MB SHM = 50% overflow
    {"name": "overflow_75pct", "num_blocks": 80, "shm_capacity_mb": 200}, # 800MB data, 200MB SHM = 75% overflow
    {"name": "overflow_90pct", "num_blocks": 100, "shm_capacity_mb": 100},# 1GB data, 100MB SHM = 90% overflow
]

###############################################################################
# REAL SHM Store using mmap on /dev/shm
###############################################################################
class RealShmStore:
    """Uses actual mmap on /dev/shm for shared memory storage."""
    
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.used = 0
        self.shm_dir = f"/dev/shm/cascade_{JOB_ID}_rank{RANK}"
        os.makedirs(self.shm_dir, exist_ok=True)
        self.files = {}  # block_id -> (path, mmap_obj, size)
    
    def put(self, block_id: str, data: bytes) -> Tuple[bool, float]:
        """Write to /dev/shm using mmap. Returns (success, time_seconds)."""
        size = len(data)
        if self.used + size > self.capacity:
            return False, 0.0  # No space in SHM
        
        start = time.perf_counter()
        path = os.path.join(self.shm_dir, f"{block_id}.bin")
        
        # Write using mmap for real SHM performance
        with open(path, 'wb') as f:
            f.write(data)
        
        # Open with mmap for reading
        fd = os.open(path, os.O_RDWR)
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        os.close(fd)
        
        self.files[block_id] = (path, mm, size)
        self.used += size
        elapsed = time.perf_counter() - start
        return True, elapsed
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        """Read from mmap-backed /dev/shm."""
        if block_id not in self.files:
            return None, 0.0
        
        path, mm, size = self.files[block_id]
        start = time.perf_counter()
        # Real mmap read
        data = mm[:]
        elapsed = time.perf_counter() - start
        return data, elapsed
    
    def clear(self):
        for block_id, (path, mm, size) in self.files.items():
            mm.close()
        shutil.rmtree(self.shm_dir, ignore_errors=True)
        self.files.clear()
        self.used = 0

###############################################################################
# Lustre Store with cold read testing
###############################################################################
class LustreStore:
    """Lustre storage with proper cold read (drop page cache)."""
    
    def __init__(self):
        self.lustre_dir = f"{SCRATCH}/lustre_{JOB_ID}/rank_{RANK:04d}"
        os.makedirs(self.lustre_dir, exist_ok=True)
        self.index = {}
    
    def put(self, block_id: str, data: bytes) -> float:
        """Write to Lustre with striping."""
        start = time.perf_counter()
        path = os.path.join(self.lustre_dir, f"{block_id}.bin")
        
        with open(path, 'wb') as f:
            f.write(data)
        os.fsync(os.open(path, os.O_RDONLY))  # Force to disk
        
        self.index[block_id] = path
        return time.perf_counter() - start
    
    def get_warm(self, block_id: str) -> Tuple[Optional[bytes], float]:
        """Read from Lustre (warm - may use page cache)."""
        if block_id not in self.index:
            return None, 0.0
        
        start = time.perf_counter()
        with open(self.index[block_id], 'rb') as f:
            data = f.read()
        return data, time.perf_counter() - start
    
    def get_cold(self, block_id: str) -> Tuple[Optional[bytes], float]:
        """Read from Lustre (cold - drop page cache before read)."""
        if block_id not in self.index:
            return None, 0.0
        
        path = self.index[block_id]
        
        # Drop page cache for this file using fadvise
        try:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except:
            pass
        
        start = time.perf_counter()
        with open(path, 'rb') as f:
            data = f.read()
        return data, time.perf_counter() - start
    
    def clear(self):
        shutil.rmtree(self.lustre_dir, ignore_errors=True)
        self.index.clear()

###############################################################################
# Tiered Cascade (SHM + Lustre spillover)
###############################################################################
class TieredCascade:
    """Real tiered store: SHM (mmap) + Lustre (spillover)."""
    
    def __init__(self, shm_capacity_bytes: int):
        self.shm = RealShmStore(shm_capacity_bytes)
        self.lustre = LustreStore()
        self.block_location = {}  # block_id -> "SHM" or "Lustre"
    
    def put(self, block_id: str, data: bytes) -> Tuple[str, float]:
        """Put to SHM first, spillover to Lustre if full."""
        success, elapsed = self.shm.put(block_id, data)
        if success:
            self.block_location[block_id] = "SHM"
            return "SHM", elapsed
        
        elapsed = self.lustre.put(block_id, data)
        self.block_location[block_id] = "Lustre"
        return "Lustre", elapsed
    
    def get(self, block_id: str, cold_lustre: bool = False) -> Tuple[Optional[bytes], str, float]:
        """Get from appropriate tier."""
        loc = self.block_location.get(block_id)
        if loc == "SHM":
            data, elapsed = self.shm.get(block_id)
            return data, "SHM", elapsed
        elif loc == "Lustre":
            if cold_lustre:
                data, elapsed = self.lustre.get_cold(block_id)
            else:
                data, elapsed = self.lustre.get_warm(block_id)
            return data, "Lustre", elapsed
        return None, "MISS", 0.0
    
    def clear(self):
        self.shm.clear()
        self.lustre.clear()
        self.block_location.clear()

###############################################################################
# LMCache Baseline (Lustre only)
###############################################################################
class LMCacheBaseline:
    """LMCache without DRAM tier - pure Lustre."""
    
    def __init__(self):
        self.lustre = LustreStore()
        self.lustre.lustre_dir = f"{SCRATCH}/lmcache_{JOB_ID}/rank_{RANK:04d}"
        os.makedirs(self.lustre.lustre_dir, exist_ok=True)
    
    def put(self, block_id: str, data: bytes) -> float:
        return self.lustre.put(block_id, data)
    
    def get_warm(self, block_id: str) -> Tuple[Optional[bytes], float]:
        return self.lustre.get_warm(block_id)
    
    def get_cold(self, block_id: str) -> Tuple[Optional[bytes], float]:
        return self.lustre.get_cold(block_id)
    
    def clear(self):
        self.lustre.clear()

###############################################################################
# Run Benchmark
###############################################################################
def run_scenario(scenario: dict) -> dict:
    name = scenario["name"]
    num_blocks = scenario["num_blocks"]
    shm_cap_bytes = scenario["shm_capacity_mb"] * 1024 * 1024
    
    if RANK == 0:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {name}")
        print(f"  Data: {num_blocks}×10MB = {num_blocks*10}MB/rank")
        print(f"  SHM capacity: {scenario['shm_capacity_mb']}MB")
        expected_overflow = max(0, (num_blocks * BLOCK_SIZE - shm_cap_bytes) / (num_blocks * BLOCK_SIZE) * 100)
        print(f"  Expected overflow: {expected_overflow:.0f}%")
        print(f"{'='*70}")
    
    # Generate test data with content-addressed hashing
    np.random.seed(42 + RANK)
    blocks = []
    for i in range(num_blocks):
        data = np.random.bytes(BLOCK_SIZE)
        block_id = hashlib.sha256(data).hexdigest()[:32]
        blocks.append((block_id, data))
    
    # ===== CASCADE WRITE =====
    cascade = TieredCascade(shm_cap_bytes)
    shm_write_times, lustre_write_times = [], []
    
    for bid, data in blocks:
        tier, t = cascade.put(bid, data)
        if tier == "SHM":
            shm_write_times.append(t)
        else:
            lustre_write_times.append(t)
    
    # ===== CASCADE READ (warm) =====
    shm_read_times, lustre_read_warm_times = [], []
    for bid, _ in blocks:
        _, tier, t = cascade.get(bid, cold_lustre=False)
        if tier == "SHM":
            shm_read_times.append(t)
        elif tier == "Lustre":
            lustre_read_warm_times.append(t)
    
    # ===== CASCADE READ (cold Lustre) =====
    lustre_read_cold_times = []
    for bid, _ in blocks:
        loc = cascade.block_location.get(bid)
        if loc == "Lustre":
            _, _, t = cascade.get(bid, cold_lustre=True)
            lustre_read_cold_times.append(t)
    
    cascade.clear()
    
    # ===== LMCACHE WRITE =====
    lmc = LMCacheBaseline()
    lmc_write_times = []
    for bid, data in blocks:
        t = lmc.put(bid, data)
        lmc_write_times.append(t)
    
    # ===== LMCACHE READ (warm - with page cache) =====
    lmc_read_warm_times = []
    for bid, _ in blocks:
        _, t = lmc.get_warm(bid)
        lmc_read_warm_times.append(t)
    
    # ===== LMCACHE READ (cold - drop page cache) =====
    lmc_read_cold_times = []
    for bid, _ in blocks:
        _, t = lmc.get_cold(bid)
        lmc_read_cold_times.append(t)
    
    lmc.clear()
    
    # Calculate bandwidth
    def calc_bw(times, n):
        if not times or sum(times) == 0:
            return 0.0
        return n * BLOCK_SIZE / sum(times) / 1e9
    
    n_shm = len(shm_write_times)
    n_lustre = len(lustre_write_times)
    overflow_pct = n_lustre / num_blocks * 100
    
    # Per-tier bandwidth
    shm_write_bw = calc_bw(shm_write_times, n_shm)
    shm_read_bw = calc_bw(shm_read_times, n_shm)
    lustre_write_bw = calc_bw(lustre_write_times, n_lustre)
    lustre_read_warm_bw = calc_bw(lustre_read_warm_times, n_lustre)
    lustre_read_cold_bw = calc_bw(lustre_read_cold_times, n_lustre)
    
    # Effective cascade bandwidth (considering tier mix)
    cascade_write_total_time = sum(shm_write_times) + sum(lustre_write_times)
    cascade_read_warm_total_time = sum(shm_read_times) + sum(lustre_read_warm_times)
    cascade_read_cold_total_time = sum(shm_read_times) + sum(lustre_read_cold_times)
    
    cascade_write_bw = num_blocks * BLOCK_SIZE / cascade_write_total_time / 1e9 if cascade_write_total_time > 0 else 0
    cascade_read_warm_bw = num_blocks * BLOCK_SIZE / cascade_read_warm_total_time / 1e9 if cascade_read_warm_total_time > 0 else 0
    cascade_read_cold_bw = num_blocks * BLOCK_SIZE / cascade_read_cold_total_time / 1e9 if cascade_read_cold_total_time > 0 else 0
    
    # LMCache bandwidth
    lmc_write_bw = calc_bw(lmc_write_times, num_blocks)
    lmc_read_warm_bw = calc_bw(lmc_read_warm_times, num_blocks)
    lmc_read_cold_bw = calc_bw(lmc_read_cold_times, num_blocks)
    
    result = {
        "scenario": name,
        "num_blocks": num_blocks,
        "n_shm": n_shm,
        "n_lustre": n_lustre,
        "overflow_pct": overflow_pct,
        # Cascade per-tier
        "cascade_shm_write_gbps": shm_write_bw,
        "cascade_shm_read_gbps": shm_read_bw,
        "cascade_lustre_write_gbps": lustre_write_bw,
        "cascade_lustre_read_warm_gbps": lustre_read_warm_bw,
        "cascade_lustre_read_cold_gbps": lustre_read_cold_bw,
        # Cascade effective
        "cascade_write_gbps": cascade_write_bw,
        "cascade_read_warm_gbps": cascade_read_warm_bw,
        "cascade_read_cold_gbps": cascade_read_cold_bw,
        # LMCache
        "lmcache_write_gbps": lmc_write_bw,
        "lmcache_read_warm_gbps": lmc_read_warm_bw,
        "lmcache_read_cold_gbps": lmc_read_cold_bw,
    }
    
    if RANK == 0:
        print(f"\n[{name}] Actual Overflow: {overflow_pct:.0f}% ({n_shm} SHM, {n_lustre} Lustre)")
        print(f"\n  CASCADE Per-Tier Bandwidth:")
        print(f"    SHM:        Write {shm_write_bw:.2f} GB/s, Read {shm_read_bw:.2f} GB/s")
        print(f"    Lustre:     Write {lustre_write_bw:.2f} GB/s")
        print(f"                Read  {lustre_read_warm_bw:.2f} GB/s (warm), {lustre_read_cold_bw:.2f} GB/s (cold)")
        print(f"\n  CASCADE Effective (mixed):")
        print(f"    Write: {cascade_write_bw:.2f} GB/s")
        print(f"    Read:  {cascade_read_warm_bw:.2f} GB/s (warm), {cascade_read_cold_bw:.2f} GB/s (cold)")
        print(f"\n  LMCACHE (Lustre only):")
        print(f"    Write: {lmc_write_bw:.2f} GB/s")
        print(f"    Read:  {lmc_read_warm_bw:.2f} GB/s (warm), {lmc_read_cold_bw:.2f} GB/s (cold)")
        
        speedup_warm = cascade_read_warm_bw / lmc_read_warm_bw if lmc_read_warm_bw > 0 else 0
        speedup_cold = cascade_read_cold_bw / lmc_read_cold_bw if lmc_read_cold_bw > 0 else 0
        print(f"\n  >>> Cascade vs LMCache: {speedup_warm:.2f}x (warm), {speedup_cold:.2f}x (cold)")
    
    return result

def main():
    if RANK == 0:
        print("\n" + "="*70)
        print("REAL TIERED BENCHMARK - mmap SHM + Lustre cold/warm read")
        print("="*70)
        print(f"Block size: {BLOCK_SIZE/1024/1024:.0f} MB")
        print(f"Ranks: {NPROCS}")
    
    results = []
    for scenario in SCENARIOS:
        r = run_scenario(scenario)
        results.append(r)
    
    # Aggregate across ranks using MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    all_results = comm.gather(results, root=0)
    
    if RANK == 0:
        # Average across ranks
        avg_results = []
        for i, scenario in enumerate(SCENARIOS):
            avg = {"scenario": scenario["name"]}
            for key in results[0].keys():
                if key != "scenario":
                    vals = [r[i][key] for r in all_results]
                    avg[key] = sum(vals) / len(vals) if vals else 0
            avg_results.append(avg)
        
        # Print summary table
        print("\n" + "="*90)
        print("AGGREGATED RESULTS (all ranks)")
        print("="*90)
        
        # Summary table
        print(f"\n{'Scenario':<18} {'Overflow':<10} {'Cascade Write':<14} {'Cascade Read':<28} {'LMCache Read':<24}")
        print(f"{'':18} {'':10} {'':14} {'Warm':<14} {'Cold':<14} {'Warm':<12} {'Cold':<12}")
        print("-"*90)
        
        for r in avg_results:
            cw = r["cascade_write_gbps"] * NPROCS
            crw = r["cascade_read_warm_gbps"] * NPROCS
            crc = r["cascade_read_cold_gbps"] * NPROCS
            lrw = r["lmcache_read_warm_gbps"] * NPROCS
            lrc = r["lmcache_read_cold_gbps"] * NPROCS
            
            print(f"{r['scenario']:<18} {r['overflow_pct']:.0f}%{'':<7} {cw:.1f} GB/s{'':<4} {crw:.1f} GB/s{'':<4} {crc:.1f} GB/s{'':<4} {lrw:.1f} GB/s{'':<4} {lrc:.1f} GB/s")
        
        print("\n" + "="*90)
        print("SPEEDUP ANALYSIS (Cascade / LMCache)")
        print("="*90)
        print(f"\n{'Scenario':<18} {'Overflow':<10} {'Read Speedup (Warm)':<22} {'Read Speedup (Cold)':<22}")
        print("-"*70)
        
        for r in avg_results:
            crw = r["cascade_read_warm_gbps"]
            crc = r["cascade_read_cold_gbps"]
            lrw = r["lmcache_read_warm_gbps"]
            lrc = r["lmcache_read_cold_gbps"]
            
            speedup_warm = crw / lrw if lrw > 0 else float('inf')
            speedup_cold = crc / lrc if lrc > 0 else float('inf')
            
            print(f"{r['scenario']:<18} {r['overflow_pct']:.0f}%{'':<7} {speedup_warm:.2f}x{'':<17} {speedup_cold:.2f}x")
        
        # Save results
        output_file = f"{PROJECT_DIR}/benchmark/results/real_tiered_{JOB_ID}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "job_id": JOB_ID,
                "ranks": NPROCS,
                "block_size_mb": BLOCK_SIZE / 1024 / 1024,
                "scenarios": SCENARIOS,
                "results": avg_results
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Cleanup
rm -rf $SCRATCH/cascade_$JOB_ID $SCRATCH/lmcache_$JOB_ID $SCRATCH/lustre_$JOB_ID
rm -rf /dev/shm/cascade_${JOB_ID}_*

echo ""
echo "[DONE] Real tiered benchmark completed."
