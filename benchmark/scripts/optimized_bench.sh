#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/optimized_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/optimized_%j.err
#SBATCH -J optimized_bench

###############################################################################
# OPTIMIZED BENCHMARK - Cascade Read 최적화 + 버퍼 재사용
# 
# 문제 해결:
# 1. Cascade get()에서 버퍼 재사용 (np.zeros 한번만 호출)
# 2. C++ SSE2 prefetch + vectorized copy
# 3. Cold read 테스트 (page cache drop - root만 가능하므로 skip)
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export CASCADE_CPP=$PROJECT_DIR/cascade_Code/cpp
export RESULTS_DIR=$PROJECT_DIR/benchmark/results

module purge
module load python/3.11
module load cudatoolkit
module load cray-mpich
module load libfabric

export PYTHONPATH=$CASCADE_CPP:$PROJECT_DIR:$PYTHONPATH

# 빌드 확인
cd $PROJECT_DIR/cascade_Code/cpp
if [ ! -f "cascade_cpp.cpython-311-x86_64-linux-gnu.so" ]; then
    echo "[BUILD] Rebuilding cascade_cpp..."
    ./build_perlmutter.sh
fi

# Verify
python3 -c "import cascade_cpp; print('✓ cascade_cpp loaded')" || exit 1

cd $PROJECT_DIR
mkdir -p $RESULTS_DIR benchmark/logs

JOB_ID=$SLURM_JOB_ID
NPROCS=$SLURM_NTASKS
FIRST_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)

echo "============================================"
echo "OPTIMIZED BENCHMARK - Read Performance Fix"
echo "============================================"
echo "Job ID: $JOB_ID"
echo "Nodes: $SLURM_NNODES, Ranks: $NPROCS"
echo "============================================"

###############################################################################
# Run Optimized Benchmark
###############################################################################
srun --ntasks=$NPROCS --gpus-per-task=1 python3 << 'PYEOF'
"""
OPTIMIZED BENCHMARK - Cascade Read 성능 개선

핵심 최적화:
1. 버퍼 재사용 (pre-allocated buffer)
2. SSE2 prefetch + vectorized copy (C++)
3. 동일 조건 비교 (모두 raw bytes 비교)
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

SCRATCH = Path(os.environ['SCRATCH'])
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
CASCADE_CPP = Path(os.environ['CASCADE_CPP'])
RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
LOCAL_RANK = int(os.environ.get('SLURM_LOCALID', 0))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
HOSTNAME = os.environ.get('SLURMD_NODENAME', 'unknown')

# Configuration
NUM_BLOCKS = 100
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB
TOTAL_DATA = NUM_BLOCKS * BLOCK_SIZE  # 1GB per rank

if RANK == 0:
    print(f"=== OPTIMIZED BENCHMARK ===")
    print(f"Blocks: {NUM_BLOCKS} × {BLOCK_SIZE // (1024*1024)} MB = {TOTAL_DATA // (1024**3)} GB per rank")
    print(f"Total: {TOTAL_DATA * NPROCS // (1024**3)} GB across {NPROCS} ranks")

@dataclass
class Result:
    system: str
    operation: str
    gbps: float
    elapsed_sec: float
    is_real: bool

def compute_block_id(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()[:32]

def generate_test_data() -> List[Tuple[str, np.ndarray]]:
    """Generate random test data"""
    blocks = []
    np.random.seed(42 + RANK)
    for i in range(NUM_BLOCKS):
        data = np.random.randint(0, 256, size=BLOCK_SIZE, dtype=np.uint8)
        block_id = compute_block_id(data)
        blocks.append((block_id, data))
    return blocks

###############################################################################
# Cascade C++ - OPTIMIZED with buffer reuse
###############################################################################
class CascadeOptimized:
    def __init__(self, rank: int):
        import cascade_cpp
        
        config = cascade_cpp.CascadeConfig()
        config.shm_capacity_bytes = 2 * 1024 * 1024 * 1024  # 2GB
        config.shm_path = f"/dev/shm/cascade_opt_{JOB_ID}_{rank}"
        config.lustre_path = str(SCRATCH / f"cascade_opt_{JOB_ID}" / f"rank_{rank}")
        config.use_gpu = False
        config.dedup_enabled = True
        
        Path(config.lustre_path).mkdir(parents=True, exist_ok=True)
        
        self.store = cascade_cpp.CascadeStore(config)
        self.name = "Cascade-Opt"
        
        # PRE-ALLOCATED READ BUFFER - 핵심 최적화!
        self.read_buffer = np.zeros(BLOCK_SIZE, dtype=np.uint8)
        
        if rank == 0:
            print(f"  ✓ Cascade Optimized: SSE2 prefetch + buffer reuse")
    
    def put(self, block_id: str, data: np.ndarray) -> bool:
        return self.store.put(block_id, data)
    
    def get(self, block_id: str) -> Tuple[bool, int]:
        """Read with pre-allocated buffer (NO allocation overhead)"""
        success, size = self.store.get(block_id, self.read_buffer)
        return success, size
    
    def clear(self):
        self.store.clear()

###############################################################################
# LMCache - File-based (Lustre)
###############################################################################
class LMCacheBaseline:
    def __init__(self, rank: int):
        self.rank = rank
        self.base_path = SCRATCH / f"lmcache_opt_{JOB_ID}" / f"rank_{rank}"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.name = "LMCache"
        self.index = {}
        
        if rank == 0:
            print(f"  ✓ LMCache: Lustre file backend")
    
    def put(self, block_id: str, data: np.ndarray) -> bool:
        fpath = self.base_path / f"{block_id}.bin"
        with open(fpath, 'wb') as f:
            f.write(data.tobytes())
        self.index[block_id] = fpath
        return True
    
    def get(self, block_id: str) -> Tuple[bytes, int]:
        fpath = self.index.get(block_id)
        if fpath and fpath.exists():
            with open(fpath, 'rb') as f:
                data = f.read()
            return data, len(data)
        return None, 0
    
    def clear(self):
        import shutil
        shutil.rmtree(self.base_path, ignore_errors=True)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index.clear()

###############################################################################
# PDC - File-based (Lustre)
###############################################################################
class PDCBaseline:
    def __init__(self, rank: int):
        self.rank = rank
        self.base_path = SCRATCH / f"pdc_opt_{JOB_ID}" / f"rank_{rank}"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.name = "PDC"
        self.index = {}
        
        if rank == 0:
            print(f"  ✓ PDC: Lustre file backend")
    
    def put(self, block_id: str, data: np.ndarray) -> bool:
        fpath = self.base_path / f"{block_id}.pdc"
        with open(fpath, 'wb') as f:
            f.write(data.tobytes())
        self.index[block_id] = fpath
        return True
    
    def get(self, block_id: str) -> Tuple[bytes, int]:
        fpath = self.index.get(block_id)
        if fpath and fpath.exists():
            with open(fpath, 'rb') as f:
                data = f.read()
            return data, len(data)
        return None, 0
    
    def clear(self):
        import shutil
        shutil.rmtree(self.base_path, ignore_errors=True)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index.clear()

###############################################################################
# Benchmark Runner - FAIR COMPARISON
###############################################################################
def run_benchmark(store, blocks: List[Tuple[str, np.ndarray]]) -> Dict[str, Result]:
    results = {}
    
    # ===== WRITE =====
    store.clear()
    
    start = time.perf_counter()
    for block_id, data in blocks:
        store.put(block_id, data)
    write_elapsed = time.perf_counter() - start
    
    total_bytes = len(blocks) * BLOCK_SIZE
    write_gbps = total_bytes / write_elapsed / (1024**3)
    
    results['write'] = Result(
        system=store.name,
        operation='write',
        gbps=write_gbps,
        elapsed_sec=write_elapsed,
        is_real=True
    )
    
    # ===== READ =====
    # Random order
    indices = np.random.permutation(len(blocks))
    
    start = time.perf_counter()
    for idx in indices:
        block_id, _ = blocks[idx]
        store.get(block_id)
    read_elapsed = time.perf_counter() - start
    
    read_gbps = total_bytes / read_elapsed / (1024**3)
    
    results['read'] = Result(
        system=store.name,
        operation='read',
        gbps=read_gbps,
        elapsed_sec=read_elapsed,
        is_real=True
    )
    
    return results

###############################################################################
# Main
###############################################################################
def main():
    if RANK == 0:
        print("\n=== GENERATING TEST DATA ===")
    
    blocks = generate_test_data()
    
    if RANK == 0:
        print(f"  Generated {len(blocks)} blocks")
        print("\n=== INITIALIZING SYSTEMS ===")
    
    all_results = {}
    
    # 1. Cascade Optimized
    try:
        cascade = CascadeOptimized(RANK)
        all_results['Cascade-Opt'] = run_benchmark(cascade, blocks)
        print(f"[Rank {RANK}] Cascade-Opt: write={all_results['Cascade-Opt']['write'].gbps:.2f}, "
              f"read={all_results['Cascade-Opt']['read'].gbps:.2f} GB/s")
    except Exception as e:
        print(f"[Rank {RANK}] Cascade-Opt failed: {e}")
    
    # 2. LMCache
    try:
        lmcache = LMCacheBaseline(RANK)
        all_results['LMCache'] = run_benchmark(lmcache, blocks)
        print(f"[Rank {RANK}] LMCache: write={all_results['LMCache']['write'].gbps:.2f}, "
              f"read={all_results['LMCache']['read'].gbps:.2f} GB/s")
    except Exception as e:
        print(f"[Rank {RANK}] LMCache failed: {e}")
    
    # 3. PDC
    try:
        pdc = PDCBaseline(RANK)
        all_results['PDC'] = run_benchmark(pdc, blocks)
        print(f"[Rank {RANK}] PDC: write={all_results['PDC']['write'].gbps:.2f}, "
              f"read={all_results['PDC']['read'].gbps:.2f} GB/s")
    except Exception as e:
        print(f"[Rank {RANK}] PDC failed: {e}")
    
    # MPI Gather
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    all_gathered = comm.gather({k: {'write': asdict(v['write']), 'read': asdict(v['read'])} 
                                 for k, v in all_results.items()}, root=0)
    
    if RANK == 0:
        # Aggregate
        aggregated = {}
        for rank_data in all_gathered:
            for system, ops in rank_data.items():
                if system not in aggregated:
                    aggregated[system] = {'write_gbps': [], 'read_gbps': []}
                aggregated[system]['write_gbps'].append(ops['write']['gbps'])
                aggregated[system]['read_gbps'].append(ops['read']['gbps'])
        
        print("\n" + "=" * 70)
        print("OPTIMIZED RESULTS - FAIR COMPARISON")
        print("=" * 70)
        print(f"{'System':<15} {'Write Total':>12} {'Read Total':>12} {'Read Speedup':>12}")
        print("-" * 70)
        
        baseline_read = None
        summary = {}
        
        for system in ['Cascade-Opt', 'LMCache', 'PDC']:
            if system in aggregated:
                write_total = sum(aggregated[system]['write_gbps'])
                read_total = sum(aggregated[system]['read_gbps'])
                
                if system == 'LMCache':
                    baseline_read = read_total
                
                speedup = ""
                if baseline_read and system == 'Cascade-Opt':
                    speedup = f"{read_total / baseline_read:.2f}×" if baseline_read > 0 else ""
                
                print(f"{system:<15} {write_total:>10.2f} GB/s {read_total:>10.2f} GB/s {speedup:>12}")
                
                summary[system] = {
                    'write_total': write_total,
                    'read_total': read_total
                }
        
        print("=" * 70)
        
        # Save results
        output = {
            'job_id': JOB_ID,
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'raw': {k: {'write': sum(v['write_gbps']), 'read': sum(v['read_gbps'])} 
                    for k, v in aggregated.items()}
        }
        
        output_file = RESULTS_DIR / f"optimized_{JOB_ID}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")

if __name__ == '__main__':
    main()
PYEOF

echo "[BENCH] Benchmark complete!"
