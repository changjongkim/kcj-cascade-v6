#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/max_debug_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/max_debug_%j.err
#SBATCH -J max_debug_bench

###############################################################################
# MAX DEBUG BENCHMARK: 4 nodes, 16 GPUs, ~200GB data
# Debug queue max: 4 nodes, 30 minutes
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export DATA_DIR=$SCRATCH/cascade_kv_cache
export RESULTS_DIR=$PROJECT_DIR/benchmark/results
export REDIS_DIR=$PROJECT_DIR/third_party/redis
export PDC_DIR=$PROJECT_DIR/third_party/pdc/install
export MERCURY_DIR=$PROJECT_DIR/third_party/mercury/install

module load python
module load cudatoolkit
module load cray-mpich
module load libfabric

export PYTHONPATH=$PROJECT_DIR/python_pkgs_py312:$PROJECT_DIR/third_party/LMCache:$PROJECT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$PDC_DIR/lib:$MERCURY_DIR/lib:/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH
export PATH=$PDC_DIR/bin:$PATH

cd $PROJECT_DIR
mkdir -p $RESULTS_DIR benchmark/logs

RANK=$SLURM_PROCID
NPROCS=$SLURM_NTASKS
HOSTNAME=$(hostname)

echo "============================================"
echo "MAX DEBUG BENCHMARK - 4 Nodes, 16 GPUs"
echo "============================================"
echo "Rank: $RANK / $NPROCS"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

###############################################################################
# Start Services on Rank 0
###############################################################################
if [ $RANK -eq 0 ]; then
    echo "[SETUP] Starting Redis..."
    REDIS_PORT=6380
    REDIS_DIR_DATA=$SCRATCH/redis_data_$$
    mkdir -p $REDIS_DIR_DATA
    
    $REDIS_DIR/src/redis-server --port $REDIS_PORT --dir $REDIS_DIR_DATA \
        --daemonize yes --maxmemory 200gb --maxmemory-policy allkeys-lru \
        --bind 0.0.0.0 --protected-mode no
    sleep 2
    echo $HOSTNAME > $SCRATCH/redis_host_$$
    
    echo "[SETUP] Starting PDC Server..."
    mkdir -p $SCRATCH/pdc_data_$$
    cd $SCRATCH/pdc_data_$$
    $PDC_DIR/bin/pdc_server &
    echo $! > $SCRATCH/pdc_pid_$$
    cd $PROJECT_DIR
    sleep 2
fi

sleep 5
REDIS_HOST=$(cat $SCRATCH/redis_host_$$ 2>/dev/null || echo "localhost")

###############################################################################
# Run Maximum Scale Benchmark
###############################################################################
echo "[BENCH] Starting benchmark on rank $RANK..."

python3 << 'PYEOF'
import os
import sys
import json
import time
import struct
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional

SCRATCH = Path(os.environ['SCRATCH'])
DATA_DIR = Path(os.environ['DATA_DIR'])
RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')

sys.path.insert(0, str(PROJECT_DIR / 'python_pkgs_py312'))
sys.path.insert(0, str(PROJECT_DIR / 'third_party/LMCache'))

# CONFIG: Maximum for 4-node debug
BLOCKS_PER_RANK = 200            # 200 blocks × 168MB = 33.6GB per rank
BLOCK_SIZE = 168 * 1024 * 1024   # 168MB per block
NUM_PREFIXES = 50                # 50 shared system prompts
UNIQUE_PER_SESSION = 4           # 4 unique per session

# Tier capacities (force Lustre overflow)
GPU_CAPACITY = 30                # 30 blocks = 5GB GPU
SHM_CAPACITY = 50                # 50 blocks = 8.4GB SHM

@dataclass
class Result:
    system: str
    rank: int
    operation: str
    num_ops: int
    total_bytes: int
    elapsed_sec: float
    throughput_gbps: float
    extra: Dict = field(default_factory=dict)

###############################################################################
# Data Generation (with prefix sharing)
###############################################################################

def generate_block(block_id: str, size: int = BLOCK_SIZE) -> Tuple[bytes, bytes]:
    """Generate KV cache block data."""
    np.random.seed(int(hashlib.md5(block_id.encode()).hexdigest()[:8], 16))
    key_size = size // 2
    value_size = size - key_size
    return np.random.bytes(key_size), np.random.bytes(value_size)

def get_block_ids() -> List[str]:
    """Get block IDs with prefix sharing pattern."""
    blocks = []
    # 50 prefixes × (50 sessions × 1 prefix ref + 4 unique) = 50 + 50×50×4 = 10050 total requests
    # But unique blocks: 50 prefixes + 50×50×4 unique = 50 + 10000 = 10050 unique in theory
    # With dedup: 50 prefixes (shared) + 4 unique per session = much fewer unique
    
    for session in range(BLOCKS_PER_RANK // (1 + UNIQUE_PER_SESSION)):
        prefix_id = f"prefix_{session % NUM_PREFIXES:03d}"
        blocks.append(prefix_id)
        
        for u in range(UNIQUE_PER_SESSION):
            unique_id = f"unique_{RANK:03d}_{session:05d}_{u:02d}"
            blocks.append(unique_id)
    
    return blocks[:BLOCKS_PER_RANK]

###############################################################################
# Cascade Store
###############################################################################

class CascadeStore:
    def __init__(self, gpu_cap=GPU_CAPACITY, shm_cap=SHM_CAPACITY):
        self.gpu = {}
        self.shm = {}
        self.lustre_path = SCRATCH / f'cascade_lustre_{RANK}'
        self.lustre_path.mkdir(parents=True, exist_ok=True)
        
        self.gpu_cap = gpu_cap
        self.shm_cap = shm_cap
        self.dedup_index = {}
        self.stats = {'dedup_hits': 0, 'gpu_writes': 0, 'shm_writes': 0, 'lustre_writes': 0}
        
        try:
            subprocess.run(['lfs', 'setstripe', '-c', '16', '-S', '4m', 
                          str(self.lustre_path)], capture_output=True)
        except: pass
    
    def _hash(self, key: bytes, value: bytes) -> str:
        h = hashlib.sha256()
        h.update(key)
        h.update(value)
        return h.hexdigest()[:32]
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        content_hash = self._hash(key, value)
        
        if content_hash in self.dedup_index:
            self.stats['dedup_hits'] += 1
            return True
        
        self.dedup_index[content_hash] = block_id
        
        if len(self.gpu) < self.gpu_cap:
            self.gpu[block_id] = (key, value)
            self.stats['gpu_writes'] += 1
            return True
        
        if len(self.shm) < self.shm_cap:
            self.shm[block_id] = (key, value)
            self.stats['shm_writes'] += 1
            return True
        
        # Lustre overflow
        fpath = self.lustre_path / f"{block_id}.bin"
        with open(fpath, 'wb') as f:
            f.write(struct.pack('<Q', len(key)))
            f.write(key)
            f.write(value)
        self.stats['lustre_writes'] += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if block_id in self.gpu:
            return self.gpu[block_id]
        if block_id in self.shm:
            return self.shm[block_id]
        fpath = self.lustre_path / f"{block_id}.bin"
        if fpath.exists():
            with open(fpath, 'rb') as f:
                key_size = struct.unpack('<Q', f.read(8))[0]
                return f.read(key_size), f.read()
        return None

###############################################################################
# vLLM Store (GPU-only with eviction)
###############################################################################

class VLLMStore:
    def __init__(self, capacity=GPU_CAPACITY):
        self.cache = {}
        self.capacity = capacity
        self.access_order = []
        self.stats = {'evictions': 0, 'writes': 0}
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        if block_id in self.cache:
            return True
        
        while len(self.cache) >= self.capacity:
            evict_id = self.access_order.pop(0)
            if evict_id in self.cache:
                del self.cache[evict_id]
                self.stats['evictions'] += 1
        
        self.cache[block_id] = (key, value)
        self.access_order.append(block_id)
        self.stats['writes'] += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        return self.cache.get(block_id)

###############################################################################
# LMCache Store (per-file Lustre)
###############################################################################

class LMCacheStore:
    def __init__(self):
        self.lustre_path = SCRATCH / f'lmcache_lustre_{RANK}'
        self.lustre_path.mkdir(parents=True, exist_ok=True)
        self.stats = {'writes': 0}
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        fpath = self.lustre_path / f"{block_id}.bin"
        with open(fpath, 'wb') as f:
            f.write(struct.pack('<Q', len(key)))
            f.write(key)
            f.write(value)
        self.stats['writes'] += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        fpath = self.lustre_path / f"{block_id}.bin"
        if fpath.exists():
            with open(fpath, 'rb') as f:
                key_size = struct.unpack('<Q', f.read(8))[0]
                return f.read(key_size), f.read()
        return None

###############################################################################
# HDF5 Store
###############################################################################

class HDF5Store:
    def __init__(self):
        import h5py
        self.fpath = SCRATCH / f'hdf5_store_{RANK}.h5'
        self.file = h5py.File(self.fpath, 'w')
        self.stats = {'writes': 0}
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        grp = self.file.create_group(block_id)
        grp.create_dataset('key', data=np.frombuffer(key, dtype=np.uint8))
        grp.create_dataset('value', data=np.frombuffer(value, dtype=np.uint8))
        self.stats['writes'] += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if block_id in self.file:
            return self.file[block_id]['key'][()].tobytes(), self.file[block_id]['value'][()].tobytes()
        return None
    
    def close(self):
        self.file.close()

###############################################################################
# Redis Store
###############################################################################

class RedisStore:
    def __init__(self, host='localhost', port=6380):
        import redis
        self.client = redis.Redis(host=host, port=port)
        self.stats = {'writes': 0}
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        self.client.set(f"{RANK}:{block_id}:key", key)
        self.client.set(f"{RANK}:{block_id}:value", value)
        self.stats['writes'] += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        key = self.client.get(f"{RANK}:{block_id}:key")
        value = self.client.get(f"{RANK}:{block_id}:value")
        if key and value:
            return key, value
        return None

###############################################################################
# Run Benchmarks
###############################################################################

def run_benchmark(store, name: str, block_ids: List[str]) -> Dict:
    print(f"[Rank {RANK}] Starting {name}...")
    
    # Write phase
    write_start = time.perf_counter()
    write_bytes = 0
    for bid in block_ids:
        key, value = generate_block(bid)
        store.put(bid, key, value)
        write_bytes += len(key) + len(value)
    write_elapsed = time.perf_counter() - write_start
    write_gbps = (write_bytes / 1e9) / write_elapsed
    
    # Read phase
    read_start = time.perf_counter()
    read_bytes = 0
    hits = 0
    for bid in block_ids:
        result = store.get(bid)
        if result:
            read_bytes += len(result[0]) + len(result[1])
            hits += 1
    read_elapsed = time.perf_counter() - read_start
    read_gbps = (read_bytes / 1e9) / read_elapsed if read_elapsed > 0 else 0
    
    result = {
        'system': name,
        'rank': RANK,
        'blocks': len(block_ids),
        'write_bytes': write_bytes,
        'write_sec': write_elapsed,
        'write_gbps': write_gbps,
        'read_bytes': read_bytes,
        'read_sec': read_elapsed,
        'read_gbps': read_gbps,
        'hit_rate': hits / len(block_ids) * 100,
        'stats': getattr(store, 'stats', {})
    }
    
    print(f"[Rank {RANK}] {name}: write={write_gbps:.2f} GB/s, read={read_gbps:.2f} GB/s, hit_rate={result['hit_rate']:.1f}%")
    return result

def main():
    block_ids = get_block_ids()
    results = {}
    
    print(f"[Rank {RANK}] Block IDs: {len(block_ids)}, unique prefixes: {len(set(b for b in block_ids if b.startswith('prefix')))}")
    
    # 1. Cascade
    cascade = CascadeStore()
    results['Cascade'] = run_benchmark(cascade, 'Cascade', block_ids)
    
    # 2. vLLM
    vllm = VLLMStore()
    results['vLLM'] = run_benchmark(vllm, 'vLLM', block_ids)
    
    # 3. LMCache
    lmcache = LMCacheStore()
    results['LMCache'] = run_benchmark(lmcache, 'LMCache', block_ids)
    
    # 4. HDF5
    try:
        hdf5 = HDF5Store()
        results['HDF5'] = run_benchmark(hdf5, 'HDF5', block_ids)
        hdf5.close()
    except Exception as e:
        print(f"HDF5 error: {e}")
        results['HDF5'] = {'error': str(e)}
    
    # 5. Redis
    try:
        import redis
        redis_store = RedisStore(host=REDIS_HOST)
        results['Redis'] = run_benchmark(redis_store, 'Redis', block_ids)
    except Exception as e:
        print(f"Redis error: {e}")
        results['Redis'] = {'error': str(e)}
    
    # Save results
    result_file = RESULTS_DIR / f'max_debug_{JOB_ID}_rank{RANK}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Rank {RANK}] Results saved to {result_file}")
    
    # Summary on rank 0
    if RANK == 0:
        print("\n" + "="*60)
        print("SUMMARY (Rank 0)")
        print("="*60)
        for sys, res in results.items():
            if isinstance(res, dict) and 'write_gbps' in res:
                print(f"{sys:12} | Write: {res['write_gbps']:6.2f} GB/s | Read: {res['read_gbps']:6.2f} GB/s | Hit: {res['hit_rate']:5.1f}%")
                if 'stats' in res:
                    print(f"             | Stats: {res['stats']}")
        print("="*60)

if __name__ == '__main__':
    main()
PYEOF

###############################################################################
# Cleanup
###############################################################################
if [ $RANK -eq 0 ]; then
    echo "[CLEANUP] Stopping services..."
    $REDIS_DIR/src/redis-cli -p 6380 shutdown 2>/dev/null || true
    kill $(cat $SCRATCH/pdc_pid_$$ 2>/dev/null) 2>/dev/null || true
    rm -f $SCRATCH/redis_host_$$ $SCRATCH/pdc_pid_$$
fi

echo "[Rank $RANK] Done."
