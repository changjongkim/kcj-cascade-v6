#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/tiered_quick_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/tiered_quick_%j.err
#SBATCH -J tiered_quick_bench

###############################################################################
# QUICK TIERED OVERFLOW BENCHMARK (debug queue)
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export CASCADE_CPP=$PROJECT_DIR/cascade_Code/cpp

module load python/3.11
module load cudatoolkit
module load cray-mpich

export PYTHONPATH=$CASCADE_CPP:$PYTHONPATH

cd $PROJECT_DIR
mkdir -p benchmark/results benchmark/logs

JOB_ID=$SLURM_JOB_ID
NPROCS=$SLURM_NTASKS

echo "============================================"
echo "QUICK TIERED OVERFLOW BENCHMARK"
echo "Job ID: $JOB_ID, Nodes: $SLURM_NNODES, Ranks: $NPROCS"
echo "============================================"

###############################################################################
# Python Benchmark Code
###############################################################################
srun python3 << 'PYTHON_SCRIPT'
import os
import sys
import time
import json
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
import shutil

RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '/pscratch/sd/s/sgkim/kcj/Cascade-kcj')

BLOCK_SIZE = 10 * 1024 * 1024  # 10MB per block

# Quick test scenarios
SCENARIOS = [
    {"name": "all_fit", "num_blocks": 20, "shm_capacity_mb": 500},  # 200MB data, 500MB SHM
    {"name": "overflow_50pct", "num_blocks": 40, "shm_capacity_mb": 200},  # 400MB data, 200MB SHM
    {"name": "overflow_90pct", "num_blocks": 100, "shm_capacity_mb": 100},  # 1GB data, 100MB SHM
]

###############################################################################
# Tiered Cascade Store
###############################################################################
class TieredCascade:
    def __init__(self, shm_capacity_bytes: int):
        self.shm_capacity = shm_capacity_bytes
        self.shm_used = 0
        self.shm_data = {}
        self.lustre_data = {}
        self.lustre_dir = f"{SCRATCH}/cascade_tiered_{JOB_ID}/rank_{RANK:04d}"
        os.makedirs(self.lustre_dir, exist_ok=True)
        self.read_buffer = bytearray(BLOCK_SIZE)
    
    def put(self, block_id: str, data: bytes) -> Tuple[str, float]:
        start = time.perf_counter()
        if self.shm_used + len(data) <= self.shm_capacity:
            self.shm_data[block_id] = data
            self.shm_used += len(data)
            return "SHM", time.perf_counter() - start
        # Overflow to Lustre
        path = os.path.join(self.lustre_dir, f"{block_id}.bin")
        with open(path, 'wb') as f:
            f.write(data)
        self.lustre_data[block_id] = path
        return "Lustre", time.perf_counter() - start
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], str, float]:
        start = time.perf_counter()
        if block_id in self.shm_data:
            return self.shm_data[block_id], "SHM", time.perf_counter() - start
        if block_id in self.lustre_data:
            with open(self.lustre_data[block_id], 'rb') as f:
                data = f.read()
            return data, "Lustre", time.perf_counter() - start
        return None, "MISS", 0.0
    
    def clear(self):
        self.shm_data.clear()
        shutil.rmtree(self.lustre_dir, ignore_errors=True)

###############################################################################
# LMCache Baseline (all Lustre)
###############################################################################
class LMCacheBaseline:
    def __init__(self):
        self.lustre_dir = f"{SCRATCH}/lmcache_tiered_{JOB_ID}/rank_{RANK:04d}"
        os.makedirs(self.lustre_dir, exist_ok=True)
        self.index = {}
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        path = os.path.join(self.lustre_dir, f"{block_id}.bin")
        with open(path, 'wb') as f:
            f.write(data)
        self.index[block_id] = path
        return time.perf_counter() - start
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        start = time.perf_counter()
        if block_id in self.index:
            with open(self.index[block_id], 'rb') as f:
                return f.read(), time.perf_counter() - start
        return None, 0.0
    
    def clear(self):
        shutil.rmtree(self.lustre_dir, ignore_errors=True)

###############################################################################
# Run Benchmark
###############################################################################
def run_scenario(scenario: dict) -> dict:
    name = scenario["name"]
    num_blocks = scenario["num_blocks"]
    shm_cap_bytes = scenario["shm_capacity_mb"] * 1024 * 1024
    
    if RANK == 0:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"  Data: {num_blocks}×10MB = {num_blocks*10}MB/rank, SHM: {scenario['shm_capacity_mb']}MB")
        print(f"{'='*60}")
    
    # Generate test data
    np.random.seed(42 + RANK)
    blocks = [(hashlib.sha256(np.random.bytes(BLOCK_SIZE)).hexdigest()[:32], np.random.bytes(BLOCK_SIZE)) for _ in range(num_blocks)]
    
    # === Cascade ===
    cascade = TieredCascade(shm_cap_bytes)
    shm_write_times, lustre_write_times = [], []
    for bid, data in blocks:
        tier, t = cascade.put(bid, data)
        (shm_write_times if tier == "SHM" else lustre_write_times).append(t)
    
    shm_read_times, lustre_read_times = [], []
    for bid, _ in blocks:
        _, tier, t = cascade.get(bid)
        if tier == "SHM":
            shm_read_times.append(t)
        elif tier == "Lustre":
            lustre_read_times.append(t)
    
    cascade.clear()
    
    # === LMCache ===
    lmc = LMCacheBaseline()
    lmc_write_times = [lmc.put(bid, data) for bid, data in blocks]
    # Drop page cache effect with fsync
    os.sync()
    lmc_read_times = [lmc.get(bid)[1] for bid, _ in blocks]
    lmc.clear()
    
    # Calculate bandwidth
    def calc_bw(times, n):
        return n * BLOCK_SIZE / sum(times) / 1e9 if times else 0
    
    shm_w_bw = calc_bw(shm_write_times, len(shm_write_times))
    lustre_w_bw = calc_bw(lustre_write_times, len(lustre_write_times))
    shm_r_bw = calc_bw(shm_read_times, len(shm_read_times))
    lustre_r_bw = calc_bw(lustre_read_times, len(lustre_read_times))
    
    cascade_total_write = calc_bw(shm_write_times + lustre_write_times, num_blocks)
    cascade_total_read = calc_bw(shm_read_times + lustre_read_times, num_blocks)
    lmc_write_bw = calc_bw(lmc_write_times, num_blocks)
    lmc_read_bw = calc_bw(lmc_read_times, num_blocks)
    
    overflow_pct = len(lustre_write_times) / num_blocks * 100
    
    result = {
        "scenario": name,
        "num_blocks": num_blocks,
        "overflow_pct": overflow_pct,
        "cascade_shm_write_gbps": shm_w_bw,
        "cascade_lustre_write_gbps": lustre_w_bw,
        "cascade_shm_read_gbps": shm_r_bw,
        "cascade_lustre_read_gbps": lustre_r_bw,
        "cascade_effective_write_gbps": cascade_total_write,
        "cascade_effective_read_gbps": cascade_total_read,
        "lmcache_write_gbps": lmc_write_bw,
        "lmcache_read_gbps": lmc_read_bw,
    }
    
    if RANK == 0:
        print(f"\n[{name}] Overflow: {overflow_pct:.0f}%")
        print(f"  Cascade SHM:    Write {shm_w_bw:.2f} GB/s, Read {shm_r_bw:.2f} GB/s ({len(shm_write_times)} blocks)")
        print(f"  Cascade Lustre: Write {lustre_w_bw:.2f} GB/s, Read {lustre_r_bw:.2f} GB/s ({len(lustre_write_times)} blocks)")
        print(f"  Cascade Total:  Write {cascade_total_write:.2f} GB/s, Read {cascade_total_read:.2f} GB/s")
        print(f"  LMCache:        Write {lmc_write_bw:.2f} GB/s, Read {lmc_read_bw:.2f} GB/s")
        print(f"  >>> Cascade vs LMCache: Write {cascade_total_write/lmc_write_bw:.1f}x, Read {cascade_total_read/lmc_read_bw:.1f}x")
    
    return result

def main():
    if RANK == 0:
        print("\n" + "="*70)
        print("TIERED OVERFLOW BENCHMARK - Quick Test")
        print("="*70)
    
    results = [run_scenario(s) for s in SCENARIOS]
    
    # Aggregate across ranks using MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    all_results = comm.gather(results, root=0)
    
    if RANK == 0:
        # Average across ranks
        avg_results = []
        for i, scenario in enumerate(SCENARIOS):
            avg = {"scenario": scenario["name"], "overflow_pct": 0}
            for key in results[0].keys():
                if key != "scenario":
                    vals = [r[i][key] for r in all_results]
                    avg[key] = sum(vals) / len(vals)
            avg_results.append(avg)
        
        # Print summary table
        print("\n" + "="*70)
        print("AGGREGATED RESULTS (16 ranks)")
        print("="*70)
        print(f"{'Scenario':<20} {'Overflow':<10} {'CascadeW':<12} {'CascadeR':<12} {'LMCacheW':<12} {'LMCacheR':<12}")
        print("-"*70)
        for r in avg_results:
            cascade_w = r["cascade_effective_write_gbps"] * NPROCS
            cascade_r = r["cascade_effective_read_gbps"] * NPROCS
            lmc_w = r["lmcache_write_gbps"] * NPROCS
            lmc_r = r["lmcache_read_gbps"] * NPROCS
            print(f"{r['scenario']:<20} {r['overflow_pct']:.0f}%{'':<6} {cascade_w:.1f}GB/s{'':<4} {cascade_r:.1f}GB/s{'':<4} {lmc_w:.1f}GB/s{'':<4} {lmc_r:.1f}GB/s")
        
        # Save
        output_file = f"{PROJECT_DIR}/benchmark/results/tiered_quick_{JOB_ID}.json"
        with open(output_file, 'w') as f:
            json.dump({"job_id": JOB_ID, "ranks": NPROCS, "results": avg_results}, f, indent=2)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Cleanup
rm -rf $SCRATCH/cascade_tiered_$JOB_ID $SCRATCH/lmcache_tiered_$JOB_ID
echo "[DONE] Quick tiered benchmark completed."
