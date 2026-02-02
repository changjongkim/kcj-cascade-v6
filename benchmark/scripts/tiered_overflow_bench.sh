#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/tiered_overflow_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/tiered_overflow_%j.err
#SBATCH -J tiered_overflow_bench

###############################################################################
# TIERED OVERFLOW BENCHMARK
# 
# Tests Cascade's REAL VALUE: What happens when data exceeds DRAM?
#
# Scenarios:
# 1. ALL-IN-MEMORY: Data << SHM capacity (baseline)
# 2. SHM-OVERFLOW:  Data > SHM capacity → Lustre eviction
# 3. LARGE-DATA:    5GB per rank exceeding 2GB SHM
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export CASCADE_CPP=$PROJECT_DIR/cascade_Code/cpp
export LMCACHE_DIR=$PROJECT_DIR/third_party/LMCache
export PDC_DIR=$PROJECT_DIR/third_party/pdc/install
export REDIS_DIR=$PROJECT_DIR/third_party/redis
export MERCURY_DIR=$PROJECT_DIR/third_party/mercury/install
export RESULTS_DIR=$PROJECT_DIR/benchmark/results

module load python/3.11
module load cudatoolkit
module load cray-mpich
module load libfabric
module load pytorch 2>/dev/null || true

export PYTHONPATH=$CASCADE_CPP:$LMCACHE_DIR:$PROJECT_DIR/python_pkgs_py312:$PYTHONPATH
export LD_LIBRARY_PATH=$PDC_DIR/lib:$MERCURY_DIR/lib:/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH
export PATH=$PDC_DIR/bin:$PATH

cd $PROJECT_DIR
mkdir -p $RESULTS_DIR benchmark/logs

JOB_ID=$SLURM_JOB_ID
NPROCS=$SLURM_NTASKS
FIRST_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)

echo "=================================================================="
echo "TIERED OVERFLOW BENCHMARK - Testing DRAM overflow to Lustre"
echo "=================================================================="
echo "Job ID: $JOB_ID"
echo "Nodes: $SLURM_NNODES, Ranks: $NPROCS"
echo "=================================================================="

###############################################################################
# Start Services
###############################################################################
if [ "$(hostname)" == "$FIRST_NODE" ]; then
    echo "[SETUP] Starting Redis..."
    REDIS_PORT=6380
    mkdir -p $SCRATCH/redis_data_$JOB_ID
    $REDIS_DIR/src/redis-server --port $REDIS_PORT --dir $SCRATCH/redis_data_$JOB_ID \
        --daemonize yes --maxmemory 100gb --bind 0.0.0.0 --protected-mode no 2>/dev/null || true
    sleep 2
    
    echo "[SETUP] Starting PDC Server..."
    mkdir -p $SCRATCH/pdc_data_$JOB_ID
    cd $SCRATCH/pdc_data_$JOB_ID
    $PDC_DIR/bin/pdc_server &
    echo $! > $SCRATCH/pdc_pid_$JOB_ID
    cd $PROJECT_DIR
    sleep 3
    
    echo $FIRST_NODE > $SCRATCH/services_host_$JOB_ID
fi
sleep 5
SERVICES_HOST=$(cat $SCRATCH/services_host_$JOB_ID 2>/dev/null || echo "localhost")

###############################################################################
# Run Tiered Overflow Benchmark
###############################################################################
echo "[BENCH] Launching tiered overflow benchmark on all $NPROCS ranks..."

srun --ntasks=$NPROCS --gpus-per-task=1 python3 << 'PYEOF'
"""
TIERED OVERFLOW BENCHMARK

Tests what happens when data exceeds DRAM capacity:
1. ALL-IN-MEMORY: Baseline with small data
2. SHM-OVERFLOW:  Force Cascade to evict from SHM to Lustre
3. LARGE-DATA:    5GB data with 2GB SHM

Key insight: Cascade shines when DRAM is insufficient and tiered storage kicks in.
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

SCRATCH = Path(os.environ['SCRATCH'])
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
CASCADE_CPP = Path(os.environ['CASCADE_CPP'])
RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
LOCAL_RANK = int(os.environ.get('SLURM_LOCALID', 0))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
HOSTNAME = os.environ.get('SLURMD_NODENAME', 'unknown')
SERVICES_HOST = open(f"{SCRATCH}/services_host_{JOB_ID}").read().strip()

print(f"[Rank {RANK}/{NPROCS}] Node: {HOSTNAME}, GPU: {LOCAL_RANK}")

###############################################################################
# EXPERIMENT CONFIGURATIONS
###############################################################################
SCENARIOS = {
    # Scenario 1: All data fits in SHM (baseline)
    "all_in_memory": {
        "shm_capacity_mb": 4096,
        "num_blocks": 100,
        "block_size_mb": 10,
        "description": "All data fits in SHM (1GB data, 4GB SHM)",
    },
    # Scenario 2: SHM intentionally small → forces Lustre overflow
    "shm_overflow": {
        "shm_capacity_mb": 256,
        "num_blocks": 100,
        "block_size_mb": 10,
        "description": "1GB data, 256MB SHM → 750MB overflows to Lustre",
    },
    # Scenario 3: Large data exceeding SHM
    "large_data": {
        "shm_capacity_mb": 2048,
        "num_blocks": 500,
        "block_size_mb": 10,
        "description": "5GB data, 2GB SHM → 3GB overflows to Lustre",
    },
}

@dataclass
class BenchmarkResult:
    scenario: str
    system: str
    rank: int
    operation: str
    num_ops: int
    total_bytes: int
    elapsed_sec: float
    throughput_gbps: float
    shm_capacity_mb: int
    data_size_mb: int
    overflow_mb: int
    shm_hit_ratio: float
    details: Dict[str, Any] = field(default_factory=dict)

###############################################################################
# Cascade C++ Store with Tiered Tracking
###############################################################################
class CascadeTieredStore:
    def __init__(self, shm_capacity_mb: int):
        self.name = "Cascade"
        self.shm_capacity_mb = shm_capacity_mb
        self.shm_bytes_written = 0
        self.lustre_bytes_written = 0
        self.store = None
        self.read_buffer = None
        
    def initialize(self) -> bool:
        try:
            sys.path.insert(0, str(CASCADE_CPP))
            import cascade_cpp
            
            config = cascade_cpp.CascadeConfig()
            config.shm_capacity_bytes = self.shm_capacity_mb * 1024 * 1024
            config.shm_path = f"/dev/shm/cascade_tiered_{JOB_ID}_{RANK}"
            lustre_dir = SCRATCH / f"cascade_tiered_{JOB_ID}" / f"rank_{RANK}"
            lustre_dir.mkdir(parents=True, exist_ok=True)
            config.lustre_path = str(lustre_dir)
            config.lustre_stripe_count = 16
            config.lustre_stripe_size = 4 * 1024 * 1024
            config.use_gpu = False
            config.gpu_device_id = LOCAL_RANK
            config.gpu_capacity_bytes = 0
            config.dedup_enabled = True
            config.compression_enabled = False
            
            self.store = cascade_cpp.CascadeStore(config)
            self.read_buffer = np.zeros(50 * 1024 * 1024, dtype=np.uint8)
            print(f"[Rank {RANK}] Cascade: SHM={self.shm_capacity_mb}MB, Lustre={lustre_dir}")
            return True
        except Exception as e:
            print(f"[Rank {RANK}] Cascade init failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def put(self, block_id: str, data: bytes) -> Tuple[float, str]:
        t0 = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        self.store.put(block_id, arr)
        elapsed = time.perf_counter() - t0
        
        if self.shm_bytes_written + len(data) <= self.shm_capacity_mb * 1024 * 1024:
            self.shm_bytes_written += len(data)
            return elapsed, "shm"
        else:
            self.lustre_bytes_written += len(data)
            return elapsed, "lustre"
    
    def get(self, block_id: str) -> Tuple[float, Optional[bytes], str]:
        t0 = time.perf_counter()
        success, actual_size = self.store.get(block_id, self.read_buffer)
        elapsed = time.perf_counter() - t0
        
        if success:
            data = self.read_buffer[:actual_size].tobytes()
            return elapsed, data, "hit"
        return elapsed, None, "miss"
    
    def get_stats(self) -> Dict:
        return {
            "shm_bytes": self.shm_bytes_written,
            "lustre_bytes": self.lustre_bytes_written,
            "shm_capacity_mb": self.shm_capacity_mb,
        }
    
    def cleanup(self):
        self.store = None
        import shutil
        shm_path = Path(f"/dev/shm/cascade_tiered_{JOB_ID}_{RANK}")
        if shm_path.exists():
            shm_path.unlink()
        lustre_dir = SCRATCH / f"cascade_tiered_{JOB_ID}" / f"rank_{RANK}"
        if lustre_dir.exists():
            shutil.rmtree(lustre_dir, ignore_errors=True)

###############################################################################
# LMCache Store (Lustre baseline)
###############################################################################
class LMCacheStore:
    def __init__(self):
        self.name = "LMCache"
        self.base_path = None
        
    def initialize(self, config) -> bool:
        try:
            self.base_path = SCRATCH / f"lmcache_tiered_{JOB_ID}" / f"rank_{RANK}"
            self.base_path.mkdir(parents=True, exist_ok=True)
            print(f"[Rank {RANK}] LMCache: path={self.base_path}")
            return True
        except Exception as e:
            print(f"[Rank {RANK}] LMCache init failed: {e}")
            return False
    
    def put(self, block_id: str, data: bytes) -> float:
        t0 = time.perf_counter()
        block_path = self.base_path / f"{block_id}.bin"
        with open(block_path, 'wb') as f:
            f.write(data)
        os.sync()
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[float, Optional[bytes]]:
        t0 = time.perf_counter()
        block_path = self.base_path / f"{block_id}.bin"
        if block_path.exists():
            with open(block_path, 'rb') as f:
                data = f.read()
            return time.perf_counter() - t0, data
        return time.perf_counter() - t0, None
    
    def cleanup(self):
        import shutil
        if self.base_path and self.base_path.exists():
            shutil.rmtree(self.base_path, ignore_errors=True)

###############################################################################
# Generate Test Data
###############################################################################
def generate_block(block_num: int, block_size: int) -> Tuple[str, bytes]:
    np.random.seed(RANK * 10000 + block_num)
    data = np.random.randint(0, 256, size=block_size, dtype=np.uint8).tobytes()
    block_id = hashlib.sha256(data).hexdigest()[:32]
    return block_id, data

###############################################################################
# Run Scenario
###############################################################################
def run_scenario(scenario_name: str, config: Dict) -> List[BenchmarkResult]:
    results = []
    
    block_size = config["block_size_mb"] * 1024 * 1024
    num_blocks = config["num_blocks"]
    shm_capacity_mb = config["shm_capacity_mb"]
    data_size_mb = num_blocks * config["block_size_mb"]
    overflow_mb = max(0, data_size_mb - shm_capacity_mb)
    
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"Config: SHM={shm_capacity_mb}MB, Data={data_size_mb}MB, Overflow={overflow_mb}MB")
    print(f"{'='*70}")
    
    blocks = [generate_block(i, block_size) for i in range(num_blocks)]
    
    # === Cascade ===
    cascade = CascadeTieredStore(shm_capacity_mb)
    if cascade.initialize():
        write_times = []
        for block_id, data in blocks:
            elapsed, tier = cascade.put(block_id, data)
            write_times.append(elapsed)
        
        total_write_time = sum(write_times)
        total_bytes = num_blocks * block_size
        write_gbps = (total_bytes / 1e9) / total_write_time if total_write_time > 0 else 0
        
        results.append(BenchmarkResult(
            scenario=scenario_name,
            system="Cascade",
            rank=RANK,
            operation="write",
            num_ops=num_blocks,
            total_bytes=total_bytes,
            elapsed_sec=total_write_time,
            throughput_gbps=write_gbps,
            shm_capacity_mb=shm_capacity_mb,
            data_size_mb=data_size_mb,
            overflow_mb=overflow_mb,
            shm_hit_ratio=min(1.0, shm_capacity_mb / data_size_mb),
            details=cascade.get_stats()
        ))
        
        read_times = []
        for block_id, _ in blocks:
            elapsed, _, _ = cascade.get(block_id)
            read_times.append(elapsed)
        
        total_read_time = sum(read_times)
        read_gbps = (total_bytes / 1e9) / total_read_time if total_read_time > 0 else 0
        
        results.append(BenchmarkResult(
            scenario=scenario_name,
            system="Cascade",
            rank=RANK,
            operation="read",
            num_ops=num_blocks,
            total_bytes=total_bytes,
            elapsed_sec=total_read_time,
            throughput_gbps=read_gbps,
            shm_capacity_mb=shm_capacity_mb,
            data_size_mb=data_size_mb,
            overflow_mb=overflow_mb,
            shm_hit_ratio=min(1.0, shm_capacity_mb / data_size_mb),
            details=cascade.get_stats()
        ))
        
        stats = cascade.get_stats()
        print(f"  [Cascade] Write: {write_gbps:.2f} GB/s, Read: {read_gbps:.2f} GB/s")
        print(f"            SHM: {stats['shm_bytes']/1e9:.2f}GB, Lustre: {stats['lustre_bytes']/1e9:.2f}GB")
        cascade.cleanup()
    
    # === LMCache ===
    lmcache = LMCacheStore()
    if lmcache.initialize(config):
        os.system("sync")
        
        write_times = []
        for block_id, data in blocks:
            elapsed = lmcache.put(block_id, data)
            write_times.append(elapsed)
        
        total_write_time = sum(write_times)
        total_bytes = num_blocks * block_size
        write_gbps = (total_bytes / 1e9) / total_write_time if total_write_time > 0 else 0
        
        results.append(BenchmarkResult(
            scenario=scenario_name,
            system="LMCache",
            rank=RANK,
            operation="write",
            num_ops=num_blocks,
            total_bytes=total_bytes,
            elapsed_sec=total_write_time,
            throughput_gbps=write_gbps,
            shm_capacity_mb=0,
            data_size_mb=data_size_mb,
            overflow_mb=data_size_mb,
            shm_hit_ratio=0.0,
            details={}
        ))
        
        time.sleep(1)
        
        read_times = []
        for block_id, _ in blocks:
            elapsed, _ = lmcache.get(block_id)
            read_times.append(elapsed)
        
        total_read_time = sum(read_times)
        read_gbps = (total_bytes / 1e9) / total_read_time if total_read_time > 0 else 0
        
        results.append(BenchmarkResult(
            scenario=scenario_name,
            system="LMCache",
            rank=RANK,
            operation="read",
            num_ops=num_blocks,
            total_bytes=total_bytes,
            elapsed_sec=total_read_time,
            throughput_gbps=read_gbps,
            shm_capacity_mb=0,
            data_size_mb=data_size_mb,
            overflow_mb=data_size_mb,
            shm_hit_ratio=0.0,
            details={}
        ))
        
        print(f"  [LMCache] Write: {write_gbps:.2f} GB/s, Read: {read_gbps:.2f} GB/s (all Lustre)")
        lmcache.cleanup()
    
    return results

###############################################################################
# Main
###############################################################################
def main():
    all_results = []
    
    for scenario_name in ["all_in_memory", "shm_overflow", "large_data"]:
        config = SCENARIOS[scenario_name]
        results = run_scenario(scenario_name, config)
        all_results.extend(results)
    
    rank_file = RESULTS_DIR / f"tiered_overflow_rank{RANK:03d}_{JOB_ID}.json"
    with open(rank_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    if RANK == 0:
        import time as time_module
        time_module.sleep(5)
        
        combined = []
        for r in range(NPROCS):
            rfile = RESULTS_DIR / f"tiered_overflow_rank{r:03d}_{JOB_ID}.json"
            if rfile.exists():
                with open(rfile) as f:
                    combined.extend(json.load(f))
        
        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(lambda: {"write": [], "read": []}))
        
        for r in combined:
            agg[r["scenario"]][r["system"]][r["operation"]].append(r["throughput_gbps"])
        
        print("\n" + "="*80)
        print("AGGREGATED RESULTS - TIERED OVERFLOW BENCHMARK")
        print(f"Nodes: {os.environ.get('SLURM_NNODES', 1)}, Ranks: {NPROCS}")
        print("="*80)
        
        for scenario in ["all_in_memory", "shm_overflow", "large_data"]:
            cfg = SCENARIOS.get(scenario, {})
            print(f"\n--- {scenario} ({cfg.get('description', '')}) ---")
            print(f"{'System':<15} | {'Write (GB/s)':<15} | {'Read (GB/s)':<15}")
            print("-" * 50)
            
            for system in ["Cascade", "LMCache"]:
                if system in agg[scenario]:
                    w_list = agg[scenario][system]["write"]
                    r_list = agg[scenario][system]["read"]
                    w_total = sum(w_list)
                    r_total = sum(r_list)
                    print(f"{system:<15} | {w_total:>12.2f}   | {r_total:>12.2f}")
        
        summary_file = RESULTS_DIR / f"tiered_overflow_summary_{JOB_ID}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "job_id": JOB_ID,
                "nprocs": NPROCS,
                "scenarios": SCENARIOS,
                "results": combined
            }, f, indent=2)
        
        print(f"\nResults saved to: {summary_file}")

if __name__ == "__main__":
    main()
PYEOF

###############################################################################
# Cleanup
###############################################################################
if [ "$(hostname)" == "$FIRST_NODE" ]; then
    echo "[CLEANUP] Stopping services..."
    $REDIS_DIR/src/redis-cli -p 6380 shutdown 2>/dev/null || true
    kill $(cat $SCRATCH/pdc_pid_$JOB_ID 2>/dev/null) 2>/dev/null || true
    rm -rf $SCRATCH/redis_data_$JOB_ID $SCRATCH/pdc_data_$JOB_ID
    rm -rf $SCRATCH/cascade_tiered_$JOB_ID $SCRATCH/lmcache_tiered_$JOB_ID
    rm -f $SCRATCH/services_host_$JOB_ID $SCRATCH/pdc_pid_$JOB_ID
fi

echo "[DONE] Tiered overflow benchmark complete"
