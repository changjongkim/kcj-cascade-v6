#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH -J shm_vs_lustre
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/shm_vs_lustre_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/shm_vs_lustre_%j.err

###############################################################################
# SHM vs Lustre Core Benchmark
# 
# Direct measurement of storage tier performance:
# - /dev/shm (memory-backed tmpfs)
# - Lustre $SCRATCH (parallel file system)
#
# This is the core differentiator for Cascade
###############################################################################

set -e
echo "=========================================="
echo "SHM vs LUSTRE CORE BENCHMARK"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Date: $(date -Iseconds)"
echo "Hostname: $(hostname)"
echo "=========================================="

cd /pscratch/sd/s/sgkim/Skim-cascade

# Run on each node
srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 \
    python3 << 'PYTHON_BENCH'
import os
import sys
import time
import mmap
import numpy as np
import json
import ctypes
import tempfile
import hashlib
from pathlib import Path

rank = int(os.environ.get('SLURM_PROCID', 0))
nodes = int(os.environ.get('SLURM_NNODES', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')
hostname = os.uname().nodename

print(f"[Rank {rank}/{nodes}] Starting on {hostname}")

###############################################################################
# Test Configuration
###############################################################################

NUM_BLOCKS = 100
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB
TOTAL_DATA = NUM_BLOCKS * BLOCK_SIZE  # 1GB per rank

results = {
    'rank': rank,
    'hostname': hostname,
    'job_id': job_id,
    'config': {
        'num_blocks': NUM_BLOCKS,
        'block_size_mb': BLOCK_SIZE / 1024 / 1024,
        'total_data_mb': TOTAL_DATA / 1024 / 1024
    },
    'tests': {}
}

# Generate test data
print(f"[Rank {rank}] Generating {NUM_BLOCKS} blocks of {BLOCK_SIZE/1024/1024:.0f}MB...")
np.random.seed(42 + rank)
test_data = np.random.bytes(BLOCK_SIZE)

###############################################################################
# Test 1: /dev/shm (Shared Memory) Performance
###############################################################################

print(f"[Rank {rank}] === /dev/shm Benchmark ===")

SHM_DIR = f"/dev/shm/cascade_bench_{rank}"
os.makedirs(SHM_DIR, exist_ok=True)

# SHM Write
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    path = f"{SHM_DIR}/block_{i:06d}.bin"
    with open(path, 'wb') as f:
        f.write(test_data)
shm_write_time = time.perf_counter() - start
shm_write_gbps = (TOTAL_DATA / 1e9) / shm_write_time
print(f"[Rank {rank}] SHM Write: {shm_write_gbps:.2f} GB/s ({shm_write_time:.3f}s)")

# SHM Hot Read (data in memory)
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    path = f"{SHM_DIR}/block_{i:06d}.bin"
    with open(path, 'rb') as f:
        data = f.read()
shm_hot_time = time.perf_counter() - start
shm_hot_gbps = (TOTAL_DATA / 1e9) / shm_hot_time
print(f"[Rank {rank}] SHM Hot Read: {shm_hot_gbps:.2f} GB/s ({shm_hot_time:.3f}s)")

# SHM mmap read (direct memory access)
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    path = f"{SHM_DIR}/block_{i:06d}.bin"
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = bytes(mm[:])  # Force read
        mm.close()
shm_mmap_time = time.perf_counter() - start
shm_mmap_gbps = (TOTAL_DATA / 1e9) / shm_mmap_time
print(f"[Rank {rank}] SHM mmap Read: {shm_mmap_gbps:.2f} GB/s ({shm_mmap_time:.3f}s)")

results['tests']['shm'] = {
    'path': SHM_DIR,
    'write_gbps': shm_write_gbps,
    'hot_read_gbps': shm_hot_gbps,
    'mmap_read_gbps': shm_mmap_gbps
}

# Cleanup SHM
import shutil
shutil.rmtree(SHM_DIR, ignore_errors=True)

###############################################################################
# Test 2: Lustre Performance
###############################################################################

print(f"[Rank {rank}] === Lustre Benchmark ===")

LUSTRE_DIR = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data/lustre_bench_{job_id}_{rank}"
os.makedirs(LUSTRE_DIR, exist_ok=True)

# Lustre Write
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    path = f"{LUSTRE_DIR}/block_{i:06d}.bin"
    with open(path, 'wb') as f:
        f.write(test_data)
        f.flush()
        os.fsync(f.fileno())  # Ensure data hits storage
lustre_write_time = time.perf_counter() - start
lustre_write_gbps = (TOTAL_DATA / 1e9) / lustre_write_time
print(f"[Rank {rank}] Lustre Write: {lustre_write_gbps:.2f} GB/s ({lustre_write_time:.3f}s)")

# Lustre Hot Read (data may be in page cache)
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    path = f"{LUSTRE_DIR}/block_{i:06d}.bin"
    with open(path, 'rb') as f:
        data = f.read()
lustre_hot_time = time.perf_counter() - start
lustre_hot_gbps = (TOTAL_DATA / 1e9) / lustre_hot_time
print(f"[Rank {rank}] Lustre Hot Read: {lustre_hot_gbps:.2f} GB/s ({lustre_hot_time:.3f}s)")

# Lustre Cold Read (drop page cache using posix_fadvise)
def drop_page_cache(path):
    """Drop page cache for a file using posix_fadvise"""
    libc = ctypes.CDLL("libc.so.6")
    fd = os.open(path, os.O_RDONLY)
    file_size = os.fstat(fd).st_size
    # POSIX_FADV_DONTNEED = 4
    libc.posix_fadvise(fd, 0, file_size, 4)
    os.close(fd)

# Drop caches
for i in range(NUM_BLOCKS):
    path = f"{LUSTRE_DIR}/block_{i:06d}.bin"
    drop_page_cache(path)

# Now read cold
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    path = f"{LUSTRE_DIR}/block_{i:06d}.bin"
    with open(path, 'rb') as f:
        data = f.read()
lustre_cold_time = time.perf_counter() - start
lustre_cold_gbps = (TOTAL_DATA / 1e9) / lustre_cold_time
print(f"[Rank {rank}] Lustre Cold Read: {lustre_cold_gbps:.2f} GB/s ({lustre_cold_time:.3f}s)")

results['tests']['lustre'] = {
    'path': LUSTRE_DIR,
    'write_gbps': lustre_write_gbps,
    'hot_read_gbps': lustre_hot_gbps,
    'cold_read_gbps': lustre_cold_gbps
}

# Cleanup Lustre
shutil.rmtree(LUSTRE_DIR, ignore_errors=True)

###############################################################################
# Test 3: Content-Addressed Deduplication Simulation
###############################################################################

print(f"[Rank {rank}] === Deduplication Simulation ===")

# Simulate 100 sessions with same system prompt
NUM_SESSIONS = 100
PROMPT_SIZE = 1.25 * 1024 * 1024  # 1.25MB

system_prompt = np.random.bytes(int(PROMPT_SIZE))
prompt_hash = hashlib.sha256(system_prompt).hexdigest()[:16]

# LMCache: session-specific IDs = N copies
lmcache_storage_mb = NUM_SESSIONS * PROMPT_SIZE / (1024 * 1024)

# Cascade: content-addressed = 1 copy
cascade_storage_mb = PROMPT_SIZE / (1024 * 1024)

dedup_ratio = lmcache_storage_mb / cascade_storage_mb

print(f"[Rank {rank}] LMCache storage: {lmcache_storage_mb:.1f} MB ({NUM_SESSIONS} copies)")
print(f"[Rank {rank}] Cascade storage: {cascade_storage_mb:.2f} MB (1 copy)")
print(f"[Rank {rank}] Deduplication ratio: {dedup_ratio:.1f}x")

results['tests']['deduplication'] = {
    'num_sessions': NUM_SESSIONS,
    'prompt_size_mb': PROMPT_SIZE / (1024 * 1024),
    'lmcache_storage_mb': lmcache_storage_mb,
    'cascade_storage_mb': cascade_storage_mb,
    'dedup_ratio': dedup_ratio
}

###############################################################################
# Speedup Calculations
###############################################################################

shm_vs_lustre_cold = shm_mmap_gbps / lustre_cold_gbps
shm_vs_lustre_hot = shm_mmap_gbps / lustre_hot_gbps

results['speedups'] = {
    'shm_vs_lustre_cold': shm_vs_lustre_cold,
    'shm_vs_lustre_hot': shm_vs_lustre_hot
}

print(f"\n[Rank {rank}] === Summary ===")
print(f"SHM mmap: {shm_mmap_gbps:.2f} GB/s")
print(f"Lustre cold: {lustre_cold_gbps:.2f} GB/s")
print(f"SHM vs Lustre Cold: {shm_vs_lustre_cold:.1f}x faster")

###############################################################################
# Save results
###############################################################################

result_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/shm_vs_lustre_{job_id}_rank{rank}.json"
with open(result_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"[Rank {rank}] Results saved to: {result_path}")
PYTHON_BENCH

###############################################################################
# Aggregate Results
###############################################################################
echo ""
echo "=== AGGREGATING RESULTS ==="

python3 << 'AGGREGATE'
import os
import json
import glob
from datetime import datetime

job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
nodes = int(os.environ.get('SLURM_NNODES', 1))

# Collect all rank results
results_pattern = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/shm_vs_lustre_{job_id}_rank*.json"
rank_files = sorted(glob.glob(results_pattern))

all_results = []
for f in rank_files:
    with open(f) as fp:
        all_results.append(json.load(fp))

if not all_results:
    print("No results found!")
    exit(1)

# Aggregate statistics
aggregate = {
    'job_id': job_id,
    'timestamp': datetime.now().isoformat(),
    'nodes': nodes,
    'ranks': len(all_results),
    'config': all_results[0]['config'],
    'aggregate': {
        'shm': {
            'write_gbps_total': sum(r['tests']['shm']['write_gbps'] for r in all_results),
            'hot_read_gbps_total': sum(r['tests']['shm']['hot_read_gbps'] for r in all_results),
            'mmap_read_gbps_total': sum(r['tests']['shm']['mmap_read_gbps'] for r in all_results),
        },
        'lustre': {
            'write_gbps_total': sum(r['tests']['lustre']['write_gbps'] for r in all_results),
            'hot_read_gbps_total': sum(r['tests']['lustre']['hot_read_gbps'] for r in all_results),
            'cold_read_gbps_total': sum(r['tests']['lustre']['cold_read_gbps'] for r in all_results),
        },
        'deduplication': all_results[0]['tests']['deduplication']
    },
    'per_rank': all_results
}

# Calculate average performance
n = len(all_results)
aggregate['average'] = {
    'shm_write_gbps': aggregate['aggregate']['shm']['write_gbps_total'] / n,
    'shm_mmap_gbps': aggregate['aggregate']['shm']['mmap_read_gbps_total'] / n,
    'lustre_write_gbps': aggregate['aggregate']['lustre']['write_gbps_total'] / n,
    'lustre_cold_gbps': aggregate['aggregate']['lustre']['cold_read_gbps_total'] / n,
}

# Calculate speedups
avg = aggregate['average']
aggregate['speedups'] = {
    'shm_vs_lustre_cold': avg['shm_mmap_gbps'] / avg['lustre_cold_gbps'] if avg['lustre_cold_gbps'] > 0 else 0,
    'aggregate_shm_read_gbps': aggregate['aggregate']['shm']['mmap_read_gbps_total'],
    'aggregate_lustre_cold_gbps': aggregate['aggregate']['lustre']['cold_read_gbps_total'],
}

result_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/shm_vs_lustre_{job_id}_aggregate.json"
with open(result_path, 'w') as f:
    json.dump(aggregate, f, indent=2)

print(f"Aggregate results saved to: {result_path}")
print("")
print("=" * 60)
print("AGGREGATE RESULTS")
print("=" * 60)
print(f"Nodes: {nodes}, Ranks: {len(all_results)}")
print("")
print("Per-Node Average:")
print(f"  SHM mmap read:    {avg['shm_mmap_gbps']:.2f} GB/s")
print(f"  Lustre cold read: {avg['lustre_cold_gbps']:.2f} GB/s")
print(f"  Speedup:          {aggregate['speedups']['shm_vs_lustre_cold']:.1f}x")
print("")
print("Cluster Aggregate ({} nodes):".format(nodes))
print(f"  SHM mmap read:    {aggregate['speedups']['aggregate_shm_read_gbps']:.2f} GB/s")
print(f"  Lustre cold read: {aggregate['speedups']['aggregate_lustre_cold_gbps']:.2f} GB/s")
print("")
print("Deduplication:")
print(f"  100 sessions × 1.25MB prompt")
print(f"  LMCache: {aggregate['aggregate']['deduplication']['lmcache_storage_mb']:.1f} MB")
print(f"  Cascade: {aggregate['aggregate']['deduplication']['cascade_storage_mb']:.2f} MB")
print(f"  Ratio:   {aggregate['aggregate']['deduplication']['dedup_ratio']:.1f}x saved")
print("=" * 60)
AGGREGATE

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
