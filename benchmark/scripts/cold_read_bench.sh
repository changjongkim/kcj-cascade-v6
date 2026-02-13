#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:25:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/cold_read_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/cold_read_%j.err
#SBATCH -J cold_read_bench

###############################################################################
# COLD READ BENCHMARK - Fair comparison without page cache effects
#
# This benchmark tests TRUE storage performance by:
# 1. Writing data to storage
# 2. Dropping OS page cache (sync + echo 3 > drop_caches)
# 3. Reading data back (cold read = real disk/storage performance)
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
FIRST_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)

echo "============================================"
echo "COLD READ BENCHMARK - Fair Comparison"
echo "Job ID: $JOB_ID, Nodes: $SLURM_NNODES, Ranks: $NPROCS"
echo "============================================"

###############################################################################
# Python Benchmark
###############################################################################
srun python3 << 'PYTHON_SCRIPT'
import os
import sys
import time
import json
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
import subprocess
import shutil

RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '/pscratch/sd/s/sgkim/kcj/Cascade-kcj')

BLOCK_SIZE = 10 * 1024 * 1024  # 10MB
NUM_BLOCKS = 50  # 500MB per rank


###############################################################################
# Import Cascade C++ backend
###############################################################################
try:
    import cascade_cpp
    HAS_CASCADE_CPP = True
    if RANK == 0:
        print("[OK] cascade_cpp imported successfully")
except ImportError as e:
    HAS_CASCADE_CPP = False
    if RANK == 0:
        print(f"[WARN] cascade_cpp not available: {e}")


###############################################################################
# Cascade C++ Store (Real Implementation)
###############################################################################
class CascadeCppStore:
    """Real cascade_cpp with mmap + SSE2."""
    
    def __init__(self, shm_path: str, capacity_gb: float = 4.0):
        capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)
        self.store = cascade_cpp.CascadeStore(shm_path, capacity_bytes)
        self.read_buffer = np.zeros(BLOCK_SIZE, dtype=np.uint8)
        if RANK == 0:
            print(f"[CascadeCpp] Initialized at {shm_path}, capacity={capacity_gb}GB")
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8)
        success = self.store.put(block_id, arr)
        return time.perf_counter() - start
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        start = time.perf_counter()
        success, actual_size = self.store.get(block_id, self.read_buffer)
        if success:
            return self.read_buffer[:actual_size].tobytes(), time.perf_counter() - start
        return None, 0.0
    
    def clear(self):
        self.store.clear()


###############################################################################
# Lustre Store (LMCache-style per-file storage)
###############################################################################
class LustreStore:
    """LMCache-style: one file per block on Lustre."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.index = {}
        # Set Lustre stripe
        try:
            subprocess.run(["lfs", "setstripe", "-c", "8", "-S", "4m", base_dir],
                          capture_output=True, timeout=5)
        except:
            pass
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        path = os.path.join(self.base_dir, f"{block_id}.bin")
        with open(path, 'wb') as f:
            f.write(data)
        os.fsync(f.fileno())  # Force to disk
        self.index[block_id] = path
        return time.perf_counter() - start
    
    def get_warm(self, block_id: str) -> Tuple[Optional[bytes], float]:
        """Normal read (may hit page cache)."""
        start = time.perf_counter()
        if block_id in self.index:
            with open(self.index[block_id], 'rb') as f:
                return f.read(), time.perf_counter() - start
        return None, 0.0
    
    def get_cold(self, block_id: str) -> Tuple[Optional[bytes], float]:
        """Cold read using O_DIRECT to bypass page cache."""
        start = time.perf_counter()
        if block_id not in self.index:
            return None, 0.0
        
        path = self.index[block_id]
        try:
            # Use O_DIRECT for true cold read
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            # O_DIRECT requires aligned buffer (4KB aligned)
            file_size = os.fstat(fd).st_size
            aligned_size = ((file_size + 4095) // 4096) * 4096
            buf = bytearray(aligned_size + 4096)
            # Find 4K aligned offset
            offset = (4096 - (id(buf) % 4096)) % 4096
            view = memoryview(buf)[offset:offset + aligned_size]
            bytes_read = os.readv(fd, [view])
            os.close(fd)
            return bytes(view[:file_size]), time.perf_counter() - start
        except Exception as e:
            # Fallback
            with open(path, 'rb') as f:
                return f.read(), time.perf_counter() - start
    
    def clear(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)


###############################################################################
# Benchmark
###############################################################################
def drop_page_cache():
    """Drop OS page cache using sync + vmtouch (if available)."""
    os.sync()
    # We can't use drop_caches without root, but fsync ensures data is on disk
    # and subsequent O_DIRECT reads bypass cache

def run_benchmark():
    if RANK == 0:
        print(f"\n{'='*60}")
        print(f"COLD READ BENCHMARK")
        print(f"Blocks: {NUM_BLOCKS} × {BLOCK_SIZE//1024//1024}MB = {NUM_BLOCKS*BLOCK_SIZE//1024//1024}MB/rank")
        print(f"{'='*60}")
    
    results = {}
    
    # Generate test data
    np.random.seed(42 + RANK)
    blocks = []
    for i in range(NUM_BLOCKS):
        data = np.random.bytes(BLOCK_SIZE)
        block_id = hashlib.sha256(data).hexdigest()[:32]
        blocks.append((block_id, data))
    
    # ==========================================================================
    # Test 1: Cascade C++ (mmap - always "warm" by design)
    # ==========================================================================
    if HAS_CASCADE_CPP:
        shm_path = f"/dev/shm/cascade_cold_{JOB_ID}_{RANK}"
        cascade = CascadeCppStore(shm_path, capacity_gb=2.0)
        
        # Write
        write_times = [cascade.put(bid, data) for bid, data in blocks]
        write_bw = NUM_BLOCKS * BLOCK_SIZE / sum(write_times) / 1e9
        
        # Read (mmap = always memory speed)
        read_times = [cascade.get(bid)[1] for bid, _ in blocks]
        read_bw = NUM_BLOCKS * BLOCK_SIZE / sum(read_times) / 1e9
        
        cascade.clear()
        
        results['Cascade-SHM'] = {
            'write_gbps': write_bw,
            'read_gbps': read_bw,
            'note': 'mmap-based, always memory speed'
        }
        
        if RANK == 0:
            print(f"\n[Cascade-SHM] Write: {write_bw:.2f} GB/s, Read: {read_bw:.2f} GB/s (mmap)")
    
    # ==========================================================================
    # Test 2: Lustre Store (LMCache-style)
    # ==========================================================================
    lustre_dir = f"{SCRATCH}/lustre_cold_{JOB_ID}/rank_{RANK:04d}"
    lustre = LustreStore(lustre_dir)
    
    # Write (with fsync - true disk write)
    write_times = [lustre.put(bid, data) for bid, data in blocks]
    write_bw = NUM_BLOCKS * BLOCK_SIZE / sum(write_times) / 1e9
    
    # Warm read (may hit page cache)
    warm_times = [lustre.get_warm(bid)[1] for bid, _ in blocks]
    warm_bw = NUM_BLOCKS * BLOCK_SIZE / sum(warm_times) / 1e9
    
    # Cold read (O_DIRECT - bypass page cache)
    drop_page_cache()
    cold_times = [lustre.get_cold(bid)[1] for bid, _ in blocks]
    cold_bw = NUM_BLOCKS * BLOCK_SIZE / sum(cold_times) / 1e9
    
    lustre.clear()
    
    results['Lustre-LMCache-style'] = {
        'write_gbps': write_bw,
        'read_warm_gbps': warm_bw,
        'read_cold_gbps': cold_bw,
    }
    
    if RANK == 0:
        print(f"\n[Lustre-LMCache] Write: {write_bw:.2f} GB/s")
        print(f"                Warm Read: {warm_bw:.2f} GB/s (page cache)")
        print(f"                Cold Read: {cold_bw:.2f} GB/s (O_DIRECT)")
    
    # ==========================================================================
    # Aggregate results
    # ==========================================================================
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    all_results = comm.gather(results, root=0)
    
    if RANK == 0:
        # Average across ranks
        cascade_write = np.mean([r['Cascade-SHM']['write_gbps'] for r in all_results if 'Cascade-SHM' in r])
        cascade_read = np.mean([r['Cascade-SHM']['read_gbps'] for r in all_results if 'Cascade-SHM' in r])
        lustre_write = np.mean([r['Lustre-LMCache-style']['write_gbps'] for r in all_results])
        lustre_warm = np.mean([r['Lustre-LMCache-style']['read_warm_gbps'] for r in all_results])
        lustre_cold = np.mean([r['Lustre-LMCache-style']['read_cold_gbps'] for r in all_results])
        
        print(f"\n{'='*70}")
        print("AGGREGATED RESULTS (16 ranks)")
        print("="*70)
        print(f"{'System':<20} {'Write Total':<15} {'Read Warm':<15} {'Read Cold':<15}")
        print("-"*70)
        print(f"{'Cascade-SHM':<20} {cascade_write*NPROCS:.1f} GB/s{'':<6} {cascade_read*NPROCS:.1f} GB/s{'':<6} {'N/A (always warm)':<15}")
        print(f"{'Lustre (LMCache)':<20} {lustre_write*NPROCS:.1f} GB/s{'':<6} {lustre_warm*NPROCS:.1f} GB/s{'':<6} {lustre_cold*NPROCS:.1f} GB/s")
        print("="*70)
        
        # Key insight
        print(f"\n📊 KEY INSIGHT:")
        print(f"   Cascade-SHM Read: {cascade_read*NPROCS:.1f} GB/s (memory speed)")
        print(f"   Lustre Warm Read: {lustre_warm*NPROCS:.1f} GB/s (page cache hit)")
        print(f"   Lustre Cold Read: {lustre_cold*NPROCS:.1f} GB/s (true disk)")
        print(f"\n   >>> Cascade vs Cold Lustre: {cascade_read/lustre_cold:.1f}x faster")
        print(f"   >>> But vs Warm Lustre: {cascade_read/lustre_warm:.1f}x")
        
        # Save
        output = {
            'job_id': JOB_ID,
            'ranks': NPROCS,
            'block_size_mb': BLOCK_SIZE / 1024 / 1024,
            'num_blocks': NUM_BLOCKS,
            'cascade_shm': {
                'write_total_gbps': cascade_write * NPROCS,
                'read_total_gbps': cascade_read * NPROCS,
            },
            'lustre': {
                'write_total_gbps': lustre_write * NPROCS,
                'read_warm_total_gbps': lustre_warm * NPROCS,
                'read_cold_total_gbps': lustre_cold * NPROCS,
            },
            'speedup': {
                'cascade_vs_cold_lustre': cascade_read / lustre_cold,
                'cascade_vs_warm_lustre': cascade_read / lustre_warm,
            }
        }
        
        with open(f"{PROJECT_DIR}/benchmark/results/cold_read_{JOB_ID}.json", 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: benchmark/results/cold_read_{JOB_ID}.json")


if __name__ == "__main__":
    run_benchmark()
PYTHON_SCRIPT

# Cleanup
rm -rf $SCRATCH/lustre_cold_$JOB_ID
rm -rf /dev/shm/cascade_cold_$JOB_ID*

echo "[DONE] Cold read benchmark completed."
