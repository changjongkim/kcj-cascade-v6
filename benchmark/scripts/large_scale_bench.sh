#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH -J large_scale
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/large_scale_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/large_scale_%j.err

###############################################################################
# LARGE SCALE BENCHMARK
# 
# Tests:
# 1. Large data volume (50GB per node)
# 2. Cross-node communication via Slingshot-11
###############################################################################

set -e
echo "=========================================="
echo "LARGE SCALE BENCHMARK"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Date: $(date -Iseconds)"
echo "=========================================="

cd /pscratch/sd/s/sgkim/Skim-cascade

###############################################################################
# Test 1: Large Data Volume (50GB per node)
###############################################################################
echo ""
echo "=== TEST 1: Large Data Volume (50GB/node) ==="

srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 \
    python3 << 'LARGE_BENCH'
import os
import time
import mmap
import numpy as np
import json

rank = int(os.environ.get('SLURM_PROCID', 0))
nodes = int(os.environ.get('SLURM_NNODES', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

# Large scale: 50GB per node
BLOCK_SIZE = 100 * 1024 * 1024  # 100MB per block
NUM_BLOCKS = 500  # 50GB total
TOTAL_DATA = NUM_BLOCKS * BLOCK_SIZE

print(f"[Rank {rank}] Testing {TOTAL_DATA/1e9:.1f}GB on /dev/shm")

# Check available SHM space
import shutil
shm_total, shm_used, shm_free = shutil.disk_usage('/dev/shm')
print(f"[Rank {rank}] /dev/shm: {shm_free/1e9:.1f}GB free of {shm_total/1e9:.1f}GB")

if shm_free < TOTAL_DATA * 1.2:
    print(f"[Rank {rank}] WARNING: Not enough SHM space, reducing to 10GB")
    NUM_BLOCKS = 100
    TOTAL_DATA = NUM_BLOCKS * BLOCK_SIZE

test_data = np.random.bytes(BLOCK_SIZE)

# SHM test
SHM_DIR = f"/dev/shm/large_bench_{rank}"
os.makedirs(SHM_DIR, exist_ok=True)

# Write
print(f"[Rank {rank}] Writing {NUM_BLOCKS} blocks of {BLOCK_SIZE/1e6:.0f}MB...")
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    with open(f"{SHM_DIR}/block_{i}.bin", 'wb') as f:
        f.write(test_data)
write_time = time.perf_counter() - start
write_gbps = (TOTAL_DATA / 1e9) / write_time
print(f"[Rank {rank}] Write: {write_gbps:.2f} GB/s ({write_time:.1f}s for {TOTAL_DATA/1e9:.1f}GB)")

# mmap read
print(f"[Rank {rank}] Reading with mmap...")
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    with open(f"{SHM_DIR}/block_{i}.bin", 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = bytes(mm[:])
        mm.close()
read_time = time.perf_counter() - start
read_gbps = (TOTAL_DATA / 1e9) / read_time
print(f"[Rank {rank}] Read: {read_gbps:.2f} GB/s ({read_time:.1f}s)")

# Save results
results = {
    'rank': rank,
    'job_id': job_id,
    'total_data_gb': TOTAL_DATA / 1e9,
    'write_gbps': write_gbps,
    'read_gbps': read_gbps,
    'write_time_s': write_time,
    'read_time_s': read_time
}

result_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/large_scale_{job_id}_rank{rank}.json"
with open(result_path, 'w') as f:
    json.dump(results, f, indent=2)

# Cleanup
shutil.rmtree(SHM_DIR, ignore_errors=True)
print(f"[Rank {rank}] Done")
LARGE_BENCH

###############################################################################
# Aggregate
###############################################################################
echo ""
echo "=== AGGREGATING RESULTS ==="

python3 << 'AGGREGATE'
import os
import json
import glob

job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
nodes = int(os.environ.get('SLURM_NNODES', 1))

pattern = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/large_scale_{job_id}_rank*.json"
files = sorted(glob.glob(pattern))

all_results = []
for f in files:
    with open(f) as fp:
        all_results.append(json.load(fp))

if not all_results:
    print("No results found!")
    exit(1)

total_data = sum(r['total_data_gb'] for r in all_results)
total_write_gbps = sum(r['write_gbps'] for r in all_results)
total_read_gbps = sum(r['read_gbps'] for r in all_results)

aggregate = {
    'job_id': job_id,
    'nodes': nodes,
    'total_data_gb': total_data,
    'aggregate_write_gbps': total_write_gbps,
    'aggregate_read_gbps': total_read_gbps,
    'avg_write_gbps': total_write_gbps / len(all_results),
    'avg_read_gbps': total_read_gbps / len(all_results),
    'per_rank': all_results
}

result_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/large_scale_{job_id}_aggregate.json"
with open(result_path, 'w') as f:
    json.dump(aggregate, f, indent=2)

print(f"Aggregate saved to: {result_path}")
print(f"")
print("=" * 60)
print("LARGE SCALE RESULTS")
print("=" * 60)
print(f"Nodes: {nodes}")
print(f"Total Data: {total_data:.1f} GB")
print(f"Aggregate Write: {total_write_gbps:.2f} GB/s")
print(f"Aggregate Read: {total_read_gbps:.2f} GB/s")
print("=" * 60)
AGGREGATE

echo ""
echo "=========================================="
echo "LARGE SCALE BENCHMARK COMPLETE"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
