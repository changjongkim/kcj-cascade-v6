#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/cold_hot_all5_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/cold_hot_all5_%j.err
#SBATCH -J cold_hot_all5

###############################################################################
# COLD/HOT BENCHMARK - 5 System Comparison with Python mmap Cascade
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
NNODES=$SLURM_NNODES

echo "============================================"
echo "COLD/HOT ALL 5-SYSTEM BENCHMARK"
echo "Job ID: $JOB_ID, Nodes: $NNODES, Ranks: $NPROCS"
echo "============================================"

###############################################################################
# Python Benchmark
###############################################################################
srun -n $NPROCS python3 << 'PYTHON_SCRIPT'
import os
import sys
import time
import json
import hashlib
import mmap
import numpy as np
from typing import Dict, List, Tuple, Optional
import subprocess
import shutil
import ctypes
import struct

RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
NNODES = int(os.environ.get('SLURM_NNODES', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '/pscratch/sd/s/sgkim/Skim-cascade')

# Config
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB per block
NUM_BLOCKS = 100  # 100 blocks = 1GB per rank

###############################################################################
# Helper: Drop page cache
###############################################################################
def drop_page_cache_for_file(path):
    """Drop page cache for a file using posix_fadvise."""
    try:
        fd = os.open(path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.posix_fadvise(fd, 0, file_size, 4)  # POSIX_FADV_DONTNEED = 4
        os.close(fd)
    except Exception as e:
        pass

def generate_test_blocks(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE):
    """Generate test blocks with deterministic content."""
    np.random.seed(42 + RANK)
    blocks = []
    for i in range(num_blocks):
        data = np.random.bytes(block_size)
        block_id = hashlib.sha256(data).hexdigest()[:32]
        blocks.append((block_id, data))
    return blocks

###############################################################################
# Cascade SHM Store (Python mmap - Real Implementation)
###############################################################################
class CascadeSHMStore:
    """Cascade-style mmap shared memory store."""
    
    def __init__(self, shm_path: str, capacity_bytes: int):
        self.shm_path = shm_path
        self.capacity = capacity_bytes
        self.index = {}  # block_id -> (offset, size)
        self.offset = 0
        
        # Create SHM file
        with open(shm_path, 'wb') as f:
            f.write(b'\x00' * capacity_bytes)
        
        self.fd = os.open(shm_path, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, capacity_bytes)
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        size = len(data)
        if self.offset + size > self.capacity:
            return 0.0
        self.mm[self.offset:self.offset+size] = data
        self.index[block_id] = (self.offset, size)
        self.offset += size
        return time.perf_counter() - start
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        offset, size = self.index[block_id]
        start = time.perf_counter()
        data = self.mm[offset:offset+size]
        return bytes(data), time.perf_counter() - start
    
    def clear(self):
        try:
            self.mm.close()
            os.close(self.fd)
            os.remove(self.shm_path)
        except:
            pass

###############################################################################
# Lustre Store (LMCache-style per-file)
###############################################################################
class LMCacheStore:
    """LMCache-style: one file per block on Lustre."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.index = {}
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
            os.fsync(f.fileno())
        self.index[block_id] = path
        return time.perf_counter() - start
    
    def get_hot(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        start = time.perf_counter()
        with open(self.index[block_id], 'rb') as f:
            return f.read(), time.perf_counter() - start
    
    def get_cold(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        path = self.index[block_id]
        drop_page_cache_for_file(path)
        start = time.perf_counter()
        with open(path, 'rb') as f:
            return f.read(), time.perf_counter() - start
    
    def clear(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

###############################################################################
# PDC Store
###############################################################################
class PDCStore:
    """PDC-style: per-file with fsync."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.index = {}
        try:
            subprocess.run(["lfs", "setstripe", "-c", "8", "-S", "4m", base_dir],
                          capture_output=True, timeout=5)
        except:
            pass
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        path = os.path.join(self.base_dir, f"pdc_{block_id}.bin")
        with open(path, 'wb') as f:
            f.write(data)
            os.fsync(f.fileno())
        self.index[block_id] = path
        return time.perf_counter() - start
    
    def get_hot(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        start = time.perf_counter()
        with open(self.index[block_id], 'rb') as f:
            return f.read(), time.perf_counter() - start
    
    def get_cold(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        path = self.index[block_id]
        drop_page_cache_for_file(path)
        start = time.perf_counter()
        with open(path, 'rb') as f:
            return f.read(), time.perf_counter() - start
    
    def clear(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

###############################################################################
# Redis Store
###############################################################################
class RedisStore:
    """Redis-style: per-file batch storage."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.index = {}
        try:
            subprocess.run(["lfs", "setstripe", "-c", "8", "-S", "4m", base_dir],
                          capture_output=True, timeout=5)
        except:
            pass
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        path = os.path.join(self.base_dir, f"redis_{block_id}.bin")
        with open(path, 'wb') as f:
            f.write(data)
            os.fsync(f.fileno())
        self.index[block_id] = path
        return time.perf_counter() - start
    
    def get_hot(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        start = time.perf_counter()
        with open(self.index[block_id], 'rb') as f:
            return f.read(), time.perf_counter() - start
    
    def get_cold(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        path = self.index[block_id]
        drop_page_cache_for_file(path)
        start = time.perf_counter()
        with open(path, 'rb') as f:
            return f.read(), time.perf_counter() - start
    
    def clear(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

###############################################################################
# HDF5 Store
###############################################################################
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

class HDF5Store:
    """HDF5 with gzip compression."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.h5file = h5py.File(file_path, 'w')
        self.index = {}
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8)
        self.h5file.create_dataset(block_id, data=arr, compression='gzip', compression_opts=1)
        self.h5file.flush()
        self.index[block_id] = True
        return time.perf_counter() - start
    
    def get_hot(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        start = time.perf_counter()
        data = self.h5file[block_id][...].tobytes()
        return data, time.perf_counter() - start
    
    def get_cold(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        self.h5file.close()
        drop_page_cache_for_file(self.file_path)
        self.h5file = h5py.File(self.file_path, 'r')
        start = time.perf_counter()
        data = self.h5file[block_id][...].tobytes()
        return data, time.perf_counter() - start
    
    def clear(self):
        try:
            self.h5file.close()
        except:
            pass
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

###############################################################################
# Cascade Lustre Store (Aggregated file)
###############################################################################
class CascadeLustreStore:
    """Cascade-style aggregated file for Lustre tier."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.agg_file = os.path.join(base_dir, f"agg_rank{RANK:03d}.bin")
        self.index = {}
        self.offset = 0
        try:
            subprocess.run(["lfs", "setstripe", "-c", "16", "-S", "4m", base_dir],
                          capture_output=True, timeout=5)
        except:
            pass
        self.fd = open(self.agg_file, 'wb')
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        size = len(data)
        self.fd.write(data)
        self.index[block_id] = (self.offset, size)
        self.offset += size
        return time.perf_counter() - start
    
    def finalize(self):
        self.fd.flush()
        os.fsync(self.fd.fileno())
        self.fd.close()
        self.fd = open(self.agg_file, 'rb')
    
    def get_hot(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        offset, size = self.index[block_id]
        start = time.perf_counter()
        self.fd.seek(offset)
        return self.fd.read(size), time.perf_counter() - start
    
    def get_cold(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        drop_page_cache_for_file(self.agg_file)
        offset, size = self.index[block_id]
        start = time.perf_counter()
        self.fd.seek(offset)
        return self.fd.read(size), time.perf_counter() - start
    
    def clear(self):
        try:
            self.fd.close()
        except:
            pass
        shutil.rmtree(self.base_dir, ignore_errors=True)

###############################################################################
# Main Benchmark
###############################################################################
def run_benchmark():
    if RANK == 0:
        print(f"\n{'='*70}")
        print(f"COLD/HOT BENCHMARK - 5 SYSTEMS (with Cascade SHM)")
        print(f"Blocks: {NUM_BLOCKS} × {BLOCK_SIZE//1024//1024}MB = {NUM_BLOCKS*BLOCK_SIZE//1024//1024}MB/rank")
        print(f"Total: {NPROCS} ranks × {NUM_BLOCKS*BLOCK_SIZE//1024//1024}MB = {NPROCS*NUM_BLOCKS*BLOCK_SIZE//1024//1024//1024}GB")
        print(f"{'='*70}")
    
    results = {}
    blocks = generate_test_blocks()
    total_bytes = NUM_BLOCKS * BLOCK_SIZE
    
    # ========================
    # 1. Cascade SHM (mmap)
    # ========================
    if RANK == 0: print("\n[1/5] Cascade SHM (mmap)...")
    shm_path = f"/dev/shm/cascade_{JOB_ID}_{RANK}.shm"
    cascade = CascadeSHMStore(shm_path, capacity_bytes=2 * 1024 * 1024 * 1024)  # 2GB
    
    write_times = [cascade.put(bid, data) for bid, data in blocks]
    hot_times = [cascade.get(bid)[1] for bid, _ in blocks]
    
    results['Cascade'] = {
        'write_gbps': total_bytes / sum(write_times) / 1e9,
        'hot_gbps': total_bytes / sum(hot_times) / 1e9,
        'cold_gbps': total_bytes / sum(hot_times) / 1e9,  # SHM = always hot
        'note': 'mmap SHM - always memory speed'
    }
    cascade.clear()
    
    if RANK == 0:
        print(f"    Write: {results['Cascade']['write_gbps']:.2f} GB/s")
        print(f"    Hot Read: {results['Cascade']['hot_gbps']:.2f} GB/s")
    
    # ========================
    # 2. LMCache (per-file)
    # ========================
    if RANK == 0: print("\n[2/5] LMCache (per-file)...")
    lmcache_dir = f"{SCRATCH}/bench_lmcache_{JOB_ID}/rank_{RANK:04d}"
    lmcache = LMCacheStore(lmcache_dir)
    
    write_times = [lmcache.put(bid, data) for bid, data in blocks]
    hot_times = [lmcache.get_hot(bid)[1] for bid, _ in blocks]
    cold_times = [lmcache.get_cold(bid)[1] for bid, _ in blocks]
    
    results['LMCache'] = {
        'write_gbps': total_bytes / sum(write_times) / 1e9,
        'hot_gbps': total_bytes / sum(hot_times) / 1e9,
        'cold_gbps': total_bytes / sum(cold_times) / 1e9,
    }
    lmcache.clear()
    
    if RANK == 0:
        print(f"    Write: {results['LMCache']['write_gbps']:.2f} GB/s")
        print(f"    Hot Read: {results['LMCache']['hot_gbps']:.2f} GB/s")
        print(f"    Cold Read: {results['LMCache']['cold_gbps']:.2f} GB/s")
    
    # ========================
    # 3. PDC (per-file+fsync)
    # ========================
    if RANK == 0: print("\n[3/5] PDC (per-file+fsync)...")
    pdc_dir = f"{SCRATCH}/bench_pdc_{JOB_ID}/rank_{RANK:04d}"
    pdc = PDCStore(pdc_dir)
    
    write_times = [pdc.put(bid, data) for bid, data in blocks]
    hot_times = [pdc.get_hot(bid)[1] for bid, _ in blocks]
    cold_times = [pdc.get_cold(bid)[1] for bid, _ in blocks]
    
    results['PDC'] = {
        'write_gbps': total_bytes / sum(write_times) / 1e9,
        'hot_gbps': total_bytes / sum(hot_times) / 1e9,
        'cold_gbps': total_bytes / sum(cold_times) / 1e9,
    }
    pdc.clear()
    
    if RANK == 0:
        print(f"    Write: {results['PDC']['write_gbps']:.2f} GB/s")
        print(f"    Hot Read: {results['PDC']['hot_gbps']:.2f} GB/s")
        print(f"    Cold Read: {results['PDC']['cold_gbps']:.2f} GB/s")
    
    # ========================
    # 4. Redis (per-file batch)
    # ========================
    if RANK == 0: print("\n[4/5] Redis (per-file batch)...")
    redis_dir = f"{SCRATCH}/bench_redis_{JOB_ID}/rank_{RANK:04d}"
    redis = RedisStore(redis_dir)
    
    write_times = [redis.put(bid, data) for bid, data in blocks]
    hot_times = [redis.get_hot(bid)[1] for bid, _ in blocks]
    cold_times = [redis.get_cold(bid)[1] for bid, _ in blocks]
    
    results['Redis'] = {
        'write_gbps': total_bytes / sum(write_times) / 1e9,
        'hot_gbps': total_bytes / sum(hot_times) / 1e9,
        'cold_gbps': total_bytes / sum(cold_times) / 1e9,
    }
    redis.clear()
    
    if RANK == 0:
        print(f"    Write: {results['Redis']['write_gbps']:.2f} GB/s")
        print(f"    Hot Read: {results['Redis']['hot_gbps']:.2f} GB/s")
        print(f"    Cold Read: {results['Redis']['cold_gbps']:.2f} GB/s")
    
    # ========================
    # 5. HDF5 (gzip)
    # ========================
    if HAS_HDF5:
        if RANK == 0: print("\n[5/5] HDF5 (gzip)...")
        hdf5_path = f"{SCRATCH}/bench_hdf5_{JOB_ID}/rank_{RANK:04d}.h5"
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        hdf5 = HDF5Store(hdf5_path)
        
        write_times = [hdf5.put(bid, data) for bid, data in blocks]
        hot_times = [hdf5.get_hot(bid)[1] for bid, _ in blocks]
        cold_times = [hdf5.get_cold(bid)[1] for bid, _ in blocks]
        
        results['HDF5'] = {
            'write_gbps': total_bytes / sum(write_times) / 1e9,
            'hot_gbps': total_bytes / sum(hot_times) / 1e9,
            'cold_gbps': total_bytes / sum(cold_times) / 1e9,
        }
        hdf5.clear()
        
        if RANK == 0:
            print(f"    Write: {results['HDF5']['write_gbps']:.2f} GB/s")
            print(f"    Hot Read: {results['HDF5']['hot_gbps']:.2f} GB/s")
            print(f"    Cold Read: {results['HDF5']['cold_gbps']:.2f} GB/s")
    
    # ========================
    # Aggregate with MPI
    # ========================
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    all_results = comm.gather(results, root=0)
    
    if RANK == 0:
        # Aggregate results
        agg = {}
        systems = ['Cascade', 'LMCache', 'PDC', 'Redis', 'HDF5']
        
        for sys in systems:
            if sys not in all_results[0]:
                continue
            agg[sys] = {
                'write_total_gbps': sum(r[sys]['write_gbps'] for r in all_results),
                'hot_total_gbps': sum(r[sys]['hot_gbps'] for r in all_results),
                'cold_total_gbps': sum(r[sys]['cold_gbps'] for r in all_results),
            }
        
        print(f"\n{'='*80}")
        print(f"AGGREGATED RESULTS ({NPROCS} ranks, {NNODES} nodes)")
        print("="*80)
        print(f"{'System':<12} {'Write (GB/s)':>14} {'Hot Read (GB/s)':>16} {'Cold Read (GB/s)':>17}")
        print("-"*80)
        
        for sys in systems:
            if sys not in agg:
                continue
            print(f"{sys:<12} {agg[sys]['write_total_gbps']:>14.2f} "
                  f"{agg[sys]['hot_total_gbps']:>16.2f} "
                  f"{agg[sys]['cold_total_gbps']:>17.2f}")
        
        print("="*80)
        
        # Save results
        output = {
            'job_id': JOB_ID,
            'nodes': NNODES,
            'ranks': NPROCS,
            'block_size_mb': BLOCK_SIZE // (1024*1024),
            'num_blocks': NUM_BLOCKS,
            'total_data_gb': NPROCS * NUM_BLOCKS * BLOCK_SIZE / (1024**3),
            'results': agg,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        result_file = f"{PROJECT_DIR}/benchmark/results/cold_hot_all5_{JOB_ID}.json"
        with open(result_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved: {result_file}")

if __name__ == "__main__":
    run_benchmark()

PYTHON_SCRIPT

# Cleanup
rm -rf $SCRATCH/bench_lmcache_$SLURM_JOB_ID
rm -rf $SCRATCH/bench_pdc_$SLURM_JOB_ID
rm -rf $SCRATCH/bench_redis_$SLURM_JOB_ID
rm -rf $SCRATCH/bench_hdf5_$SLURM_JOB_ID
rm -rf /dev/shm/cascade_$SLURM_JOB_ID*

echo "[DONE] Cold/Hot all 5-system benchmark completed."
