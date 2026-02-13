#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/real_4node_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/real_4node_%j.err
#SBATCH -J real_4node_bench

###############################################################################
# REAL 4-NODE BENCHMARK: Uses actual storage systems, NO simulation
# 
# - 4 nodes × 4 GPUs = 16 ranks
# - All storage systems are REAL (Lustre, HDF5, Redis, SHM, GPU)
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export RESULTS_DIR=$PROJECT_DIR/benchmark/results
export REDIS_DIR=$PROJECT_DIR/third_party/redis

module load python
module load cudatoolkit
module load cray-mpich

export PYTHONPATH=$PROJECT_DIR/python_pkgs_py312:$PROJECT_DIR:$PYTHONPATH

cd $PROJECT_DIR
mkdir -p $RESULTS_DIR benchmark/logs

JOB_ID=$SLURM_JOB_ID
NPROCS=$SLURM_NTASKS

echo "============================================"
echo "REAL 4-NODE BENCHMARK"
echo "============================================"
echo "Job ID: $JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Total Ranks: $NPROCS"
echo "============================================"

###############################################################################
# Start Redis on first node only
###############################################################################
FIRST_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
echo "[SETUP] First node: $FIRST_NODE"

if [ "$(hostname)" == "$FIRST_NODE" ]; then
    echo "[SETUP] Starting Redis on $FIRST_NODE..."
    REDIS_PORT=6380
    REDIS_DATA=$SCRATCH/redis_real_$JOB_ID
    mkdir -p $REDIS_DATA
    
    $REDIS_DIR/src/redis-server --port $REDIS_PORT --dir $REDIS_DATA \
        --daemonize yes --maxmemory 100gb --maxmemory-policy allkeys-lru \
        --bind 0.0.0.0 --protected-mode no 2>/dev/null || true
    sleep 2
    
    if $REDIS_DIR/src/redis-cli -p $REDIS_PORT ping 2>/dev/null | grep -q PONG; then
        echo "[SETUP] Redis OK on $FIRST_NODE"
    fi
fi

# Write Redis host for all ranks
echo $FIRST_NODE > $SCRATCH/redis_host_$JOB_ID
sleep 3

###############################################################################
# Run benchmark on all ranks using srun
###############################################################################
echo "[BENCH] Launching benchmarks on all $NPROCS ranks..."

srun --ntasks=$NPROCS --gpus-per-task=1 python3 << 'PYEOF'
"""
REAL BENCHMARK: No simulation, actual storage system I/O
Runs on ALL ranks via srun
"""
import os
import sys
import json
import time
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

SCRATCH = Path(os.environ['SCRATCH'])
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
LOCAL_RANK = int(os.environ.get('SLURM_LOCALID', 0))
HOSTNAME = os.environ.get('SLURMD_NODENAME', 'unknown')

# Read Redis host
REDIS_HOST = open(f"{SCRATCH}/redis_host_{JOB_ID}").read().strip()

# Config: 100 blocks × 10MB = 1GB per rank, 16GB total
NUM_BLOCKS = 100
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB
TOTAL_DATA_PER_RANK = NUM_BLOCKS * BLOCK_SIZE

print(f"[Rank {RANK}/{NPROCS}] Node: {HOSTNAME}, GPU: {LOCAL_RANK}, "
      f"Blocks: {NUM_BLOCKS}, Size: {TOTAL_DATA_PER_RANK//1024//1024}MB")

@dataclass
class BenchmarkResult:
    system: str
    rank: int
    operation: str
    num_ops: int
    total_bytes: int
    elapsed_sec: float
    throughput_gbps: float
    avg_latency_ms: float
    is_real: bool
    details: Dict[str, Any] = field(default_factory=dict)

###############################################################################
# REAL Storage Backends
###############################################################################

class LustrePerFileStore:
    """REAL Lustre per-file I/O"""
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.is_real = True
        self.name = "Lustre-PerFile"
        
    def put(self, block_id: str, data: bytes) -> float:
        fpath = self.base_path / f"{block_id}.bin"
        t0 = time.perf_counter()
        with open(fpath, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        fpath = self.base_path / f"{block_id}.bin"
        t0 = time.perf_counter()
        if fpath.exists():
            with open(fpath, 'rb') as f:
                data = f.read()
            return data, time.perf_counter() - t0
        return None, time.perf_counter() - t0
    
    def clear(self):
        import shutil
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)


class LustreAggregatedStore:
    """REAL Lustre aggregated I/O with striping"""
    def __init__(self, base_path: Path, blocks_per_file: int = 10):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.blocks_per_file = blocks_per_file
        self.is_real = True
        self.name = "Lustre-Aggregated"
        self.index = {}
        self.current_file_id = 0
        self.blocks_in_current = 0
        
        try:
            subprocess.run(['lfs', 'setstripe', '-c', '16', '-S', '4m', 
                          str(self.base_path)], capture_output=True, timeout=5)
        except: pass
    
    def put(self, block_id: str, data: bytes) -> float:
        t0 = time.perf_counter()
        
        if self.blocks_in_current >= self.blocks_per_file:
            self.current_file_id += 1
            self.blocks_in_current = 0
        
        fpath = self.base_path / f"agg_{RANK:03d}_{self.current_file_id:06d}.bin"
        with open(fpath, 'ab') as f:
            offset = f.tell()
            f.write(data)
            self.index[block_id] = (self.current_file_id, offset, len(data))
        
        self.blocks_in_current += 1
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        t0 = time.perf_counter()
        if block_id not in self.index:
            return None, time.perf_counter() - t0
        
        file_id, offset, size = self.index[block_id]
        fpath = self.base_path / f"agg_{RANK:03d}_{file_id:06d}.bin"
        with open(fpath, 'rb') as f:
            f.seek(offset)
            data = f.read(size)
        return data, time.perf_counter() - t0
    
    def flush(self):
        pass
    
    def clear(self):
        import shutil
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.current_file_id = 0
        self.blocks_in_current = 0


class HDF5Store:
    """REAL HDF5 storage via h5py"""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_real = True
        self.name = "HDF5"
        self.h5file = None
        
    def initialize(self):
        import h5py
        self.h5file = h5py.File(str(self.file_path), 'w')
        self.h5file.create_group('blocks')
        return True
    
    def put(self, block_id: str, data: bytes) -> float:
        t0 = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8)
        self.h5file['blocks'].create_dataset(block_id, data=arr, compression='gzip', compression_opts=1)
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        t0 = time.perf_counter()
        if block_id in self.h5file['blocks']:
            data = self.h5file['blocks'][block_id][:].tobytes()
            return data, time.perf_counter() - t0
        return None, time.perf_counter() - t0
    
    def flush(self):
        self.h5file.flush()
    
    def close(self):
        if self.h5file:
            self.h5file.close()


class RedisStore:
    """REAL Redis storage via redis-py"""
    def __init__(self, host: str, port: int = 6380):
        self.host = host
        self.port = port
        self.is_real = True
        self.name = "Redis"
        self.client = None
        
    def initialize(self) -> bool:
        try:
            import redis
            self.client = redis.Redis(host=self.host, port=self.port, socket_timeout=10)
            self.client.ping()
            return True
        except Exception as e:
            print(f"[Rank {RANK}] Redis connect failed: {e}")
            return False
    
    def put(self, block_id: str, data: bytes) -> float:
        t0 = time.perf_counter()
        self.client.set(f"r{RANK}:{block_id}", data)
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        t0 = time.perf_counter()
        data = self.client.get(f"r{RANK}:{block_id}")
        return data, time.perf_counter() - t0
    
    def clear(self):
        for key in self.client.scan_iter(f"r{RANK}:*"):
            self.client.delete(key)


class SharedMemoryStore:
    """REAL shared memory via /dev/shm"""
    def __init__(self):
        self.base_path = Path(f"/dev/shm/cascade_bench_{RANK}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.is_real = True
        self.name = "SharedMemory"
        
    def put(self, block_id: str, data: bytes) -> float:
        fpath = self.base_path / f"{block_id}.bin"
        t0 = time.perf_counter()
        with open(fpath, 'wb') as f:
            f.write(data)
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        fpath = self.base_path / f"{block_id}.bin"
        t0 = time.perf_counter()
        if fpath.exists():
            with open(fpath, 'rb') as f:
                data = f.read()
            return data, time.perf_counter() - t0
        return None, time.perf_counter() - t0
    
    def clear(self):
        import shutil
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)


class GPUMemoryStore:
    """REAL GPU memory via CuPy"""
    def __init__(self):
        self.is_real = True
        self.name = "GPU-Memory"
        self.cache = {}
        self.available = False
        
    def initialize(self) -> bool:
        try:
            import cupy as cp
            gpu_id = LOCAL_RANK
            cp.cuda.Device(gpu_id).use()
            self.cp = cp
            self.available = True
            print(f"[Rank {RANK}] Using GPU {gpu_id}")
            return True
        except Exception as e:
            print(f"[Rank {RANK}] CuPy not available: {e}")
            return False
    
    def put(self, block_id: str, data: bytes) -> float:
        if not self.available:
            return 0.0
        t0 = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8)
        self.cache[block_id] = self.cp.asarray(arr)
        self.cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if not self.available:
            return None, 0.0
        t0 = time.perf_counter()
        if block_id in self.cache:
            data = self.cp.asnumpy(self.cache[block_id]).tobytes()
            self.cp.cuda.Stream.null.synchronize()
            return data, time.perf_counter() - t0
        return None, time.perf_counter() - t0
    
    def clear(self):
        self.cache = {}
        if self.available:
            self.cp.get_default_memory_pool().free_all_blocks()


###############################################################################
# Benchmark Runner
###############################################################################

def generate_test_data() -> List[Tuple[str, bytes]]:
    blocks = []
    for i in range(NUM_BLOCKS):
        np.random.seed(RANK * 10000 + i)
        data = np.random.bytes(BLOCK_SIZE)
        block_id = hashlib.sha256(data).hexdigest()[:16]
        blocks.append((block_id, data))
    return blocks


def run_benchmark(store, blocks: List[Tuple[str, bytes]]) -> Dict[str, BenchmarkResult]:
    results = {}
    
    # WRITE
    write_latencies = []
    total_written = 0
    t0 = time.perf_counter()
    for block_id, data in blocks:
        latency = store.put(block_id, data)
        write_latencies.append(latency * 1000)
        total_written += len(data)
    if hasattr(store, 'flush'):
        store.flush()
    write_elapsed = time.perf_counter() - t0
    write_gbps = (total_written / 1e9) / write_elapsed if write_elapsed > 0 else 0
    
    results['write'] = BenchmarkResult(
        system=store.name, rank=RANK, operation='write',
        num_ops=len(blocks), total_bytes=total_written,
        elapsed_sec=write_elapsed, throughput_gbps=write_gbps,
        avg_latency_ms=np.mean(write_latencies),
        is_real=store.is_real,
        details={'p50': np.percentile(write_latencies, 50),
                 'p99': np.percentile(write_latencies, 99)}
    )
    
    # READ (random order)
    read_latencies = []
    total_read = 0
    hits = 0
    indices = np.random.permutation(len(blocks))
    
    t0 = time.perf_counter()
    for idx in indices:
        block_id, _ = blocks[idx]
        data, latency = store.get(block_id)
        read_latencies.append(latency * 1000)
        if data is not None:
            total_read += len(data)
            hits += 1
    read_elapsed = time.perf_counter() - t0
    read_gbps = (total_read / 1e9) / read_elapsed if read_elapsed > 0 else 0
    
    results['read'] = BenchmarkResult(
        system=store.name, rank=RANK, operation='read',
        num_ops=len(blocks), total_bytes=total_read,
        elapsed_sec=read_elapsed, throughput_gbps=read_gbps,
        avg_latency_ms=np.mean(read_latencies),
        is_real=store.is_real,
        details={'hits': hits, 'hit_rate': hits/len(blocks),
                 'p50': np.percentile(read_latencies, 50),
                 'p99': np.percentile(read_latencies, 99)}
    )
    
    return results


def main():
    print(f"[Rank {RANK}] Generating test data...")
    blocks = generate_test_data()
    
    all_results = {}
    
    # 1. Lustre Per-File
    print(f"[Rank {RANK}] Testing Lustre-PerFile...")
    lustre_pf = LustrePerFileStore(SCRATCH / f"bench_lustre_pf_{JOB_ID}" / f"rank_{RANK}")
    lustre_pf.clear()
    all_results['Lustre-PerFile'] = run_benchmark(lustre_pf, blocks)
    
    # 2. Lustre Aggregated
    print(f"[Rank {RANK}] Testing Lustre-Aggregated...")
    lustre_agg = LustreAggregatedStore(SCRATCH / f"bench_lustre_agg_{JOB_ID}" / f"rank_{RANK}")
    lustre_agg.clear()
    all_results['Lustre-Aggregated'] = run_benchmark(lustre_agg, blocks)
    
    # 3. HDF5
    print(f"[Rank {RANK}] Testing HDF5...")
    try:
        hdf5 = HDF5Store(SCRATCH / f"bench_hdf5_{JOB_ID}" / f"rank_{RANK}.h5")
        hdf5.initialize()
        all_results['HDF5'] = run_benchmark(hdf5, blocks)
        hdf5.close()
    except Exception as e:
        print(f"[Rank {RANK}] HDF5 failed: {e}")
    
    # 4. Redis
    print(f"[Rank {RANK}] Testing Redis...")
    redis_store = RedisStore(REDIS_HOST, 6380)
    if redis_store.initialize():
        redis_store.clear()
        all_results['Redis'] = run_benchmark(redis_store, blocks)
    
    # 5. Shared Memory
    print(f"[Rank {RANK}] Testing SharedMemory...")
    shm = SharedMemoryStore()
    shm.clear()
    all_results['SharedMemory'] = run_benchmark(shm, blocks)
    
    # 6. GPU Memory
    print(f"[Rank {RANK}] Testing GPU-Memory...")
    gpu = GPUMemoryStore()
    if gpu.initialize():
        all_results['GPU-Memory'] = run_benchmark(gpu, blocks)
        gpu.clear()
    
    # Save results
    output = {
        'metadata': {
            'job_id': JOB_ID, 'rank': RANK, 'nprocs': NPROCS,
            'hostname': HOSTNAME, 'gpu_id': LOCAL_RANK,
            'timestamp': datetime.now().isoformat(),
            'num_blocks': NUM_BLOCKS, 'block_size_mb': BLOCK_SIZE / 1024 / 1024,
        },
        'results': {k: {'write': asdict(v['write']), 'read': asdict(v['read'])} 
                   for k, v in all_results.items()}
    }
    
    result_file = RESULTS_DIR / f"real_4node_{JOB_ID}_rank{RANK:02d}.json"
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n[Rank {RANK}] ===== SUMMARY (ALL REAL) =====")
    for sys_name, r in all_results.items():
        w, rd = r['write'].throughput_gbps, r['read'].throughput_gbps
        print(f"  {sys_name:<20}: Write {w:.3f} GB/s, Read {rd:.3f} GB/s")
    
    # Cleanup
    shm.clear()


if __name__ == '__main__':
    main()

PYEOF

echo "[Rank 0] Benchmarks completed on all ranks."

###############################################################################
# Aggregate results on first node
###############################################################################
sleep 10  # Wait for all ranks

if [ "$(hostname)" == "$FIRST_NODE" ]; then
    echo ""
    echo "============================================"
    echo "AGGREGATING RESULTS FROM ALL RANKS"
    echo "============================================"
    
    python3 << 'AGGEOF'
import json
from pathlib import Path
import os
import numpy as np

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

# Collect all rank results
all_ranks = []
for f in sorted(RESULTS_DIR.glob(f"real_4node_{JOB_ID}_rank*.json")):
    with open(f) as fp:
        all_ranks.append(json.load(fp))

if not all_ranks:
    print("No results found!")
    exit(1)

print(f"Found {len(all_ranks)} rank results")

# Aggregate
systems = list(all_ranks[0]['results'].keys())
aggregated = {
    'metadata': {
        'job_id': JOB_ID,
        'num_ranks': len(all_ranks),
        'num_nodes': 4,
        'total_data_gb': len(all_ranks) * 1.0,  # 1GB per rank
    },
    'per_rank': {},
    'aggregated': {}
}

for sys in systems:
    write_gbps = [r['results'][sys]['write']['throughput_gbps'] for r in all_ranks if sys in r['results']]
    read_gbps = [r['results'][sys]['read']['throughput_gbps'] for r in all_ranks if sys in r['results']]
    
    if write_gbps:
        aggregated['aggregated'][sys] = {
            'write_gbps_per_rank_mean': np.mean(write_gbps),
            'write_gbps_total': sum(write_gbps),
            'read_gbps_per_rank_mean': np.mean(read_gbps),
            'read_gbps_total': sum(read_gbps),
            'num_ranks': len(write_gbps),
            'is_real': True
        }

# Save
agg_file = RESULTS_DIR / f"real_4node_{JOB_ID}_aggregated.json"
with open(agg_file, 'w') as f:
    json.dump(aggregated, f, indent=2)

print("")
print("=" * 75)
print(f"AGGREGATED RESULTS - {len(all_ranks)} RANKS, 4 NODES - ALL REAL")
print("=" * 75)
print(f"{'System':<20} | {'Write/Rank':>12} | {'Write Total':>12} | {'Read/Rank':>12} | {'Read Total':>12}")
print("-" * 75)
for sys, vals in aggregated['aggregated'].items():
    print(f"{sys:<20} | {vals['write_gbps_per_rank_mean']:>10.3f}GB/s | {vals['write_gbps_total']:>10.2f}GB/s | {vals['read_gbps_per_rank_mean']:>10.3f}GB/s | {vals['read_gbps_total']:>10.2f}GB/s")
print("=" * 75)
print(f"\nResults saved to: {agg_file}")

AGGEOF

    # Stop Redis
    echo "[CLEANUP] Stopping Redis..."
    $REDIS_DIR/src/redis-cli -h $FIRST_NODE -p 6380 shutdown 2>/dev/null || true
    
    # Cleanup temp data
    rm -rf $SCRATCH/bench_lustre_pf_$JOB_ID 2>/dev/null || true
    rm -rf $SCRATCH/bench_lustre_agg_$JOB_ID 2>/dev/null || true
    rm -rf $SCRATCH/bench_hdf5_$JOB_ID 2>/dev/null || true
    rm -f $SCRATCH/redis_host_$JOB_ID 2>/dev/null || true
fi

echo "Done."
