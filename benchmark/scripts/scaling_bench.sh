#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH -J scaling_bench
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/scaling_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/scaling_%j.err

###############################################################################
# SCALING BENCHMARK
# 
# Tests:
# 1. Block size scaling (1MB, 10MB, 50MB, 100MB)
# 2. Data volume scaling (1GB, 5GB, 10GB per node)
# 3. Multi-rank per node (1, 2, 4 ranks)
###############################################################################

set -e
echo "=========================================="
echo "SCALING BENCHMARK"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Date: $(date -Iseconds)"
echo "=========================================="

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj

###############################################################################
# Test 1: Block Size Scaling
###############################################################################
echo ""
echo "=== TEST 1: Block Size Scaling ==="

for BLOCK_MB in 1 10 50 100; do
    echo ""
    echo "--- Block Size: ${BLOCK_MB} MB ---"
    
    srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 \
        python3 - $BLOCK_MB << 'BLOCK_BENCH'
import os
import sys
import time
import mmap
import numpy as np

block_mb = int(sys.argv[1])
rank = int(os.environ.get('SLURM_PROCID', 0))

BLOCK_SIZE = block_mb * 1024 * 1024
NUM_BLOCKS = max(10, 1000 // block_mb)  # Total ~1GB
TOTAL_DATA = NUM_BLOCKS * BLOCK_SIZE

test_data = np.random.bytes(BLOCK_SIZE)

# SHM test
SHM_DIR = f"/dev/shm/block_bench_{rank}"
os.makedirs(SHM_DIR, exist_ok=True)

# Write
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    with open(f"{SHM_DIR}/block_{i}.bin", 'wb') as f:
        f.write(test_data)
write_time = time.perf_counter() - start
write_gbps = (TOTAL_DATA / 1e9) / write_time

# mmap read
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    with open(f"{SHM_DIR}/block_{i}.bin", 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = bytes(mm[:])
        mm.close()
read_time = time.perf_counter() - start
read_gbps = (TOTAL_DATA / 1e9) / read_time

print(f"[Rank {rank}] Block {block_mb}MB: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")

# Cleanup
import shutil
shutil.rmtree(SHM_DIR, ignore_errors=True)
BLOCK_BENCH
done

###############################################################################
# Test 2: Data Volume Scaling on SHM
###############################################################################
echo ""
echo "=== TEST 2: Data Volume Scaling ==="

for DATA_GB in 1 5 10; do
    echo ""
    echo "--- Data Volume: ${DATA_GB} GB per node ---"
    
    srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 \
        python3 - $DATA_GB << 'VOLUME_BENCH'
import os
import sys
import time
import mmap
import numpy as np

data_gb = int(sys.argv[1])
rank = int(os.environ.get('SLURM_PROCID', 0))

BLOCK_SIZE = 10 * 1024 * 1024  # 10MB
NUM_BLOCKS = (data_gb * 1024) // 10  # Number of 10MB blocks
TOTAL_DATA = NUM_BLOCKS * BLOCK_SIZE

test_data = np.random.bytes(BLOCK_SIZE)

# SHM test
SHM_DIR = f"/dev/shm/volume_bench_{rank}"
os.makedirs(SHM_DIR, exist_ok=True)

# Write
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    with open(f"{SHM_DIR}/block_{i}.bin", 'wb') as f:
        f.write(test_data)
write_time = time.perf_counter() - start
write_gbps = (TOTAL_DATA / 1e9) / write_time

# mmap read
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    with open(f"{SHM_DIR}/block_{i}.bin", 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = bytes(mm[:])
        mm.close()
read_time = time.perf_counter() - start
read_gbps = (TOTAL_DATA / 1e9) / read_time

print(f"[Rank {rank}] {data_gb}GB: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s (total: {TOTAL_DATA/1e9:.1f}GB)")

# Cleanup
import shutil
shutil.rmtree(SHM_DIR, ignore_errors=True)
VOLUME_BENCH
done

###############################################################################
# Test 3: Multi-Rank Per Node
###############################################################################
echo ""
echo "=== TEST 3: Multi-Rank Scaling (per node) ==="

for RANKS_PER_NODE in 1 2 4; do
    TOTAL_RANKS=$((SLURM_NNODES * RANKS_PER_NODE))
    echo ""
    echo "--- ${RANKS_PER_NODE} rank(s) per node, ${TOTAL_RANKS} total ---"
    
    srun -N $SLURM_NNODES -n $TOTAL_RANKS --ntasks-per-node=$RANKS_PER_NODE \
        python3 << 'RANK_BENCH'
import os
import time
import mmap
import numpy as np
import json

rank = int(os.environ.get('SLURM_PROCID', 0))

BLOCK_SIZE = 10 * 1024 * 1024
NUM_BLOCKS = 100
TOTAL_DATA = NUM_BLOCKS * BLOCK_SIZE

test_data = np.random.bytes(BLOCK_SIZE)

# SHM test (each rank has its own directory)
SHM_DIR = f"/dev/shm/rank_bench_{rank}"
os.makedirs(SHM_DIR, exist_ok=True)

# Write
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    with open(f"{SHM_DIR}/block_{i}.bin", 'wb') as f:
        f.write(test_data)
write_time = time.perf_counter() - start
write_gbps = (TOTAL_DATA / 1e9) / write_time

# Read
start = time.perf_counter()
for i in range(NUM_BLOCKS):
    with open(f"{SHM_DIR}/block_{i}.bin", 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = bytes(mm[:])
        mm.close()
read_time = time.perf_counter() - start
read_gbps = (TOTAL_DATA / 1e9) / read_time

print(f"[Rank {rank}] Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")

# Cleanup
import shutil
shutil.rmtree(SHM_DIR, ignore_errors=True)
RANK_BENCH
done

###############################################################################
# Aggregate Summary
###############################################################################
echo ""
echo "=========================================="
echo "SCALING BENCHMARK COMPLETE"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
