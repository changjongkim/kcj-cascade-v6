#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J cascade_v5_bench
#SBATCH --gpus-per-node=1
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v5_bench_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/v5_bench_%j.err

# ============================================================================
# Cascade V5 Comprehensive Benchmark
#
# Tests:
#   Part 1: Raw backend throughput (GPU, SHM, Lustre) — via C++ cascade_bench
#   Part 2: New V5 features (LRU eviction, O_DIRECT, semantic eviction,
#           tier promotion/demotion) — via Python cascade_cpp
#   Part 3: Integrated CascadeStore throughput (3-tier with eviction)
# ============================================================================

set -e

# Load modules
module load python cudatoolkit cmake gcc/12.2.0

export PYTHONPATH=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build:$PYTHONPATH

# Paths
PROJECT_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj
BUILD_DIR=$PROJECT_DIR/cascade_Code/cpp/build
RESULT_DIR=$PROJECT_DIR/benchmark/results
LUSTRE_BENCH_DIR=$PROJECT_DIR/benchmark/data/lustre_bench_v5_${SLURM_JOB_ID}

mkdir -p $RESULT_DIR
mkdir -p $LUSTRE_BENCH_DIR

JOB_ID=${SLURM_JOB_ID:-local}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE=$RESULT_DIR/v5_bench_${JOB_ID}_${TIMESTAMP}.json

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Cascade V5 Comprehensive Benchmark                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Job ID:    ${JOB_ID}                                   "
echo "║  Node:      $(hostname)                                 "
echo "║  Timestamp: ${TIMESTAMP}                                "
echo "║  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# ============================================================================
# Part 1: Raw Backend Throughput (C++ Benchmark)
# ============================================================================
echo "═══════════════════════════════════════════════════════════"
echo "  Part 1: Raw Backend Throughput (cascade_bench)"
echo "═══════════════════════════════════════════════════════════"
echo

echo "--- Test 1a: GPU + SHM (128KB blocks x 1000, 8 threads) ---"
$BUILD_DIR/cascade_bench --blocks 1000 --size 128 --threads 8 --iterations 3 2>&1
echo

echo "--- Test 1b: GPU + SHM (1MB blocks x 500, 8 threads) ---"
$BUILD_DIR/cascade_bench --blocks 500 --size 1024 --threads 8 --iterations 3 2>&1
echo

echo "--- Test 1c: GPU + SHM + Lustre (128KB x 500, 8 threads) ---"
$BUILD_DIR/cascade_bench --blocks 500 --size 128 --threads 8 --iterations 3 \
    --lustre --lustre-path $LUSTRE_BENCH_DIR 2>&1
echo

# Clean up Lustre test data between runs
rm -rf $LUSTRE_BENCH_DIR/*


# ============================================================================
# Part 2+3: Python V5 Feature Tests + Integrated CascadeStore Throughput
# ============================================================================
echo "═══════════════════════════════════════════════════════════"
echo "  Part 2+3: V5 Features & CascadeStore Throughput (Python)"
echo "═══════════════════════════════════════════════════════════"

python3 << 'PYEOF'
import numpy as np
import cascade_cpp
import time
import json
import os
import shutil

# ============================================================================
# Configuration
# ============================================================================
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
TIMESTAMP = os.environ.get("TIMESTAMP", "unknown")
RESULT_FILE = os.environ.get("RESULT_FILE", "/tmp/cascade_v5_result.json")
LUSTRE_DIR = os.environ.get("LUSTRE_BENCH_DIR", "/tmp/cascade_v5_lustre")

results = {
    "job_id": JOB_ID,
    "timestamp": TIMESTAMP,
    "node": os.environ.get("SLURMD_NODENAME", "unknown"),
    "version": "V5",
    "tests": {}
}

def timed_throughput(func, total_bytes, label=""):
    """Run func, return GB/s"""
    t0 = time.perf_counter()
    result = func()
    dt = time.perf_counter() - t0
    gbps = total_bytes / dt / (1024**3)
    print(f"  {label}: {gbps:.2f} GB/s ({dt*1000:.1f}ms)")
    return gbps, dt, result

# ============================================================================
# Test 2a: SHM Backend — write/read throughput
# ============================================================================
print()
print("═" * 60)
print("  Test 2a: SHM Backend (mmap + SSE2 + free-list)")
print("═" * 60)

NUM_BLOCKS = 100
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB — same as prior benchmark
TOTAL_DATA = NUM_BLOCKS * BLOCK_SIZE

# Generate data
blocks_data = []
block_ids = []
for i in range(NUM_BLOCKS):
    d = np.random.randint(0, 255, BLOCK_SIZE, dtype=np.uint8)
    bid = cascade_cpp.compute_block_id(d)
    blocks_data.append(d)
    block_ids.append(bid)

# SHM write
cfg_shm = cascade_cpp.CascadeConfig()
cfg_shm.use_gpu = False
cfg_shm.shm_capacity_bytes = int(TOTAL_DATA * 1.2)
cfg_shm.lustre_path = ""
cfg_shm.dedup_enabled = False

shm = cascade_cpp.ShmBackend(int(TOTAL_DATA * 1.2), "/dev/shm/cascade_v5_bench")

def shm_write():
    for i in range(NUM_BLOCKS):
        shm.put(block_ids[i], blocks_data[i])
    return True

def shm_read():
    out = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    for i in range(NUM_BLOCKS):
        shm.get(block_ids[i], out)
    return True

write_gbps, _, _ = timed_throughput(shm_write, TOTAL_DATA, "SHM Write")
read_gbps, _, _ = timed_throughput(shm_read, TOTAL_DATA, "SHM Read ")

results["tests"]["shm_raw"] = {
    "num_blocks": NUM_BLOCKS,
    "block_size_mb": BLOCK_SIZE / (1024*1024),
    "write_gbps": write_gbps,
    "read_gbps": read_gbps
}

shm.clear()

# ============================================================================
# Test 2b: Lustre O_DIRECT — write/read throughput
# ============================================================================
print()
print("═" * 60)
print("  Test 2b: Lustre Backend (O_DIRECT)")
print("═" * 60)

lustre_dir = LUSTRE_DIR + "_py"
os.makedirs(lustre_dir, exist_ok=True)
lustre = cascade_cpp.LustreBackend(lustre_dir, 4 * 1024 * 1024, 4)

# Use fewer blocks for Lustre (slower)
L_BLOCKS = 50
L_SIZE = 10 * 1024 * 1024
L_TOTAL = L_BLOCKS * L_SIZE

l_data = []
l_ids = []
for i in range(L_BLOCKS):
    d = np.random.randint(0, 255, L_SIZE, dtype=np.uint8)
    bid = cascade_cpp.compute_block_id(d)
    l_data.append(d)
    l_ids.append(bid)

def lustre_write():
    for i in range(L_BLOCKS):
        lustre.put(l_ids[i], l_data[i])
    lustre.flush()
    return True

def lustre_read():
    out = np.zeros(L_SIZE + 4096, dtype=np.uint8)
    for i in range(L_BLOCKS):
        lustre.get(l_ids[i], out)
    return True

l_write_gbps, _, _ = timed_throughput(lustre_write, L_TOTAL, "Lustre O_DIRECT Write")
l_read_gbps, _, _ = timed_throughput(lustre_read, L_TOTAL, "Lustre O_DIRECT Read ")

results["tests"]["lustre_odirect"] = {
    "num_blocks": L_BLOCKS,
    "block_size_mb": L_SIZE / (1024*1024),
    "write_gbps": l_write_gbps,
    "read_gbps": l_read_gbps
}

shutil.rmtree(lustre_dir, ignore_errors=True)

# ============================================================================
# Test 2c: LRU Eviction Throughput
# ============================================================================
print()
print("═" * 60)
print("  Test 2c: LRU Eviction (SHM → Lustre demotion)")
print("═" * 60)

EVICT_SHM_CAP = 50 * 1024 * 1024  # 50MB SHM — will overflow
evict_lustre_dir = LUSTRE_DIR + "_evict"
os.makedirs(evict_lustre_dir, exist_ok=True)

cfg_evict = cascade_cpp.CascadeConfig()
cfg_evict.use_gpu = False
cfg_evict.shm_capacity_bytes = EVICT_SHM_CAP
cfg_evict.lustre_path = evict_lustre_dir
cfg_evict.dedup_enabled = False
cfg_evict.semantic_eviction = False
cfg_evict.promotion_enabled = True

store_evict = cascade_cpp.CascadeStore(cfg_evict)

# Put 100 x 10MB = 1GB into 50MB SHM → will trigger ~950MB of evictions
E_BLOCKS = 100
E_SIZE = 10 * 1024 * 1024
E_TOTAL = E_BLOCKS * E_SIZE

e_data = []
e_ids = []
for i in range(E_BLOCKS):
    d = np.random.randint(0, 255, E_SIZE, dtype=np.uint8)
    bid = cascade_cpp.compute_block_id(d)
    e_data.append(d)
    e_ids.append(bid)

def evict_write():
    for i in range(E_BLOCKS):
        store_evict.put(e_ids[i], e_data[i], False)
    return True

evict_write_gbps, evict_dt, _ = timed_throughput(evict_write, E_TOTAL, "Write (with eviction)")

stats = store_evict.get_stats()
print(f"  SHM used: {stats.shm_used / (1024*1024):.0f}MB")
print(f"  SHM evictions: {stats.shm_evictions}")
print(f"  Eviction rate: {stats.shm_evictions / evict_dt:.0f} evictions/sec")

# Read back — some from SHM (hot), rest from Lustre (cold)
def evict_read():
    out = np.zeros(E_SIZE + 4096, dtype=np.uint8)
    found = 0
    for i in range(E_BLOCKS):
        ok, sz = store_evict.get(e_ids[i], out)
        if ok:
            found += 1
    return found

evict_read_gbps, _, found = timed_throughput(evict_read, E_TOTAL, "Read (SHM+Lustre+promote)")

stats2 = store_evict.get_stats()
print(f"  Found {found}/{E_BLOCKS} blocks")
print(f"  SHM hits: {stats2.shm_hits}, Lustre hits: {stats2.lustre_hits}")
print(f"  Promotions to SHM: {stats2.promotions_to_shm}")

results["tests"]["lru_eviction"] = {
    "shm_capacity_mb": EVICT_SHM_CAP / (1024*1024),
    "total_data_mb": E_TOTAL / (1024*1024),
    "write_gbps": evict_write_gbps,
    "read_gbps": evict_read_gbps,
    "shm_evictions": stats2.shm_evictions,
    "shm_hits": stats2.shm_hits,
    "lustre_hits": stats2.lustre_hits,
    "promotions_to_shm": stats2.promotions_to_shm,
    "blocks_found": found,
    "blocks_total": E_BLOCKS
}

shutil.rmtree(evict_lustre_dir, ignore_errors=True)

# ============================================================================
# Test 2d: Semantic Eviction (prefix preservation)
# ============================================================================
print()
print("═" * 60)
print("  Test 2d: Semantic Eviction (prefix block preservation)")
print("═" * 60)

sem_lustre_dir = LUSTRE_DIR + "_semantic"
os.makedirs(sem_lustre_dir, exist_ok=True)

SEM_SHM_CAP = 30 * 1024 * 1024  # 30MB

cfg_sem = cascade_cpp.CascadeConfig()
cfg_sem.use_gpu = False
cfg_sem.shm_capacity_bytes = SEM_SHM_CAP
cfg_sem.lustre_path = sem_lustre_dir
cfg_sem.dedup_enabled = False
cfg_sem.semantic_eviction = True
cfg_sem.promotion_enabled = False

store_sem = cascade_cpp.CascadeStore(cfg_sem)

# Put 3 prefix blocks (10MB each = 30MB, fills SHM)
prefix_ids = []
prefix_data = []
for i in range(3):
    d = np.random.randint(0, 255, E_SIZE, dtype=np.uint8)
    bid = cascade_cpp.compute_block_id(d)
    store_sem.put(bid, d, True)  # is_prefix=True
    prefix_ids.append(bid)
    prefix_data.append(d)
print(f"  Put 3 prefix blocks (30MB) → SHM full")

# Put 10 unique blocks (10MB each) → forces eviction
unique_ids = []
for i in range(10):
    d = np.random.randint(0, 255, E_SIZE, dtype=np.uint8)
    bid = cascade_cpp.compute_block_id(d)
    store_sem.put(bid, d, False)  # is_prefix=False
    unique_ids.append(bid)

stats_sem = store_sem.get_stats()
print(f"  After 10 unique block puts:")
print(f"    SHM evictions: {stats_sem.shm_evictions}")

# Check: prefix blocks should still be in SHM
prefix_in_shm = 0
out = np.zeros(E_SIZE + 4096, dtype=np.uint8)
for pid in prefix_ids:
    found, sz = store_sem.get(pid, out)
    if found:
        prefix_in_shm += 1

stats_sem2 = store_sem.get_stats()
print(f"  Prefix blocks still accessible: {prefix_in_shm}/{len(prefix_ids)}")
print(f"  SHM hits (prefix reads): {stats_sem2.shm_hits}")
print(f"  Lustre hits: {stats_sem2.lustre_hits}")

prefix_preserved = prefix_in_shm == len(prefix_ids) and stats_sem2.shm_hits >= len(prefix_ids)
print(f"  Prefix preservation: {'PASSED ✓' if prefix_preserved else 'PARTIAL'}")

results["tests"]["semantic_eviction"] = {
    "shm_capacity_mb": SEM_SHM_CAP / (1024*1024),
    "prefix_blocks": len(prefix_ids),
    "unique_blocks": 10,
    "shm_evictions": stats_sem2.shm_evictions,
    "prefix_preserved": prefix_in_shm,
    "prefix_total": len(prefix_ids),
    "pass": prefix_preserved
}

shutil.rmtree(sem_lustre_dir, ignore_errors=True)

# ============================================================================
# Test 3: Full CascadeStore Integrated Throughput
# ============================================================================
print()
print("═" * 60)
print("  Test 3: Integrated CascadeStore (SHM + Lustre + eviction)")
print("═" * 60)

store_lustre_dir = LUSTRE_DIR + "_integrated"
os.makedirs(store_lustre_dir, exist_ok=True)

# Realistic config: 128MB SHM, Lustre overflow
cfg_full = cascade_cpp.CascadeConfig()
cfg_full.use_gpu = False
cfg_full.shm_capacity_bytes = 128 * 1024 * 1024  # 128MB
cfg_full.lustre_path = store_lustre_dir
cfg_full.dedup_enabled = True
cfg_full.semantic_eviction = True
cfg_full.promotion_enabled = True

store_full = cascade_cpp.CascadeStore(cfg_full)

# Phase 1: Write 250 unique blocks x 2MB = 500MB (overflow SHM)
F_BLOCKS = 250
F_SIZE = 2 * 1024 * 1024  # 2MB
F_TOTAL = F_BLOCKS * F_SIZE

f_data = []
f_ids = []
for i in range(F_BLOCKS):
    d = np.random.randint(0, 255, F_SIZE, dtype=np.uint8)
    bid = cascade_cpp.compute_block_id(d)
    f_data.append(d)
    f_ids.append(bid)

print(f"  Config: 128MB SHM + Lustre, {F_BLOCKS} x {F_SIZE//1024}KB = {F_TOTAL//(1024*1024)}MB total")

def full_write():
    for i in range(F_BLOCKS):
        store_full.put(f_ids[i], f_data[i], i < 10)  # first 10 are prefix
    return True

def full_read():
    out = np.zeros(F_SIZE + 4096, dtype=np.uint8)
    found = 0
    for i in range(F_BLOCKS):
        ok, sz = store_full.get(f_ids[i], out)
        if ok:
            found += 1
    return found

# Write phase
fw_gbps, _, _ = timed_throughput(full_write, F_TOTAL, "Store write (500MB → 128MB SHM)")

sf = store_full.get_stats()
print(f"    SHM used: {sf.shm_used/(1024*1024):.0f}MB, evictions: {sf.shm_evictions}")
print(f"    Dedup hits: {sf.dedup_hits}")

# Read phase (mix of SHM hot + Lustre cold + promotion)
fr_gbps, _, found = timed_throughput(full_read, F_TOTAL, "Store read  (SHM+Lustre+promote)")

sf2 = store_full.get_stats()
print(f"    Found: {found}/{F_BLOCKS}")
print(f"    SHM hits: {sf2.shm_hits}, Lustre hits: {sf2.lustre_hits}")
print(f"    Promotions to SHM: {sf2.promotions_to_shm}")

# Phase 2: Dedup test — write same blocks again
print()
t0 = time.perf_counter()
for i in range(F_BLOCKS):
    store_full.put(f_ids[i], f_data[i], False)
dedup_dt = time.perf_counter() - t0
dedup_gbps = F_TOTAL / dedup_dt / (1024**3)
sf3 = store_full.get_stats()
print(f"  Dedup write (same data): {dedup_gbps:.2f} GB/s ({dedup_dt*1000:.1f}ms)")
print(f"    Dedup hits: {sf3.dedup_hits}")

results["tests"]["integrated_store"] = {
    "shm_capacity_mb": 128,
    "total_data_mb": F_TOTAL / (1024*1024),
    "num_blocks": F_BLOCKS,
    "block_size_kb": F_SIZE // 1024,
    "write_gbps": fw_gbps,
    "read_gbps": fr_gbps,
    "dedup_write_gbps": dedup_gbps,
    "shm_evictions": sf2.shm_evictions,
    "shm_hits": sf2.shm_hits,
    "lustre_hits": sf2.lustre_hits,
    "promotions_to_shm": sf2.promotions_to_shm,
    "dedup_hits": sf3.dedup_hits,
    "blocks_found": found,
    "blocks_total": F_BLOCKS
}

shutil.rmtree(store_lustre_dir, ignore_errors=True)

# ============================================================================
# Comparison with prior results
# ============================================================================
print()
print("═" * 60)
print("  Comparison with Prior Results")
print("═" * 60)

# Prior results from job 48439256 (4-node aggregate)
prior = {
    "shm_write_gbps": 3.11,
    "shm_read_gbps": 8.61,
    "lustre_write_gbps": 0.64,
    "lustre_cold_gbps": 1.09
}

cur_shm_w = results["tests"]["shm_raw"]["write_gbps"]
cur_shm_r = results["tests"]["shm_raw"]["read_gbps"]
cur_lus_w = results["tests"]["lustre_odirect"]["write_gbps"]
cur_lus_r = results["tests"]["lustre_odirect"]["read_gbps"]

print(f"  {'Metric':<25} {'Prior (V3)':>12} {'Current (V5)':>12} {'Change':>10}")
print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*10}")
print(f"  {'SHM Write (GB/s)':<25} {prior['shm_write_gbps']:>12.2f} {cur_shm_w:>12.2f} {(cur_shm_w/prior['shm_write_gbps']-1)*100:>+9.1f}%")
print(f"  {'SHM Read (GB/s)':<25} {prior['shm_read_gbps']:>12.2f} {cur_shm_r:>12.2f} {(cur_shm_r/prior['shm_read_gbps']-1)*100:>+9.1f}%")
print(f"  {'Lustre Write (GB/s)':<25} {prior['lustre_write_gbps']:>12.2f} {cur_lus_w:>12.2f} {(cur_lus_w/prior['lustre_write_gbps']-1)*100:>+9.1f}%")
print(f"  {'Lustre Read (GB/s)':<25} {prior['lustre_cold_gbps']:>12.2f} {cur_lus_r:>12.2f} {(cur_lus_r/prior['lustre_cold_gbps']-1)*100:>+9.1f}%")

results["comparison"] = {
    "prior_job_id": "48439256",
    "prior_version": "V3",
    "shm_write_change_pct": (cur_shm_w / prior["shm_write_gbps"] - 1) * 100,
    "shm_read_change_pct": (cur_shm_r / prior["shm_read_gbps"] - 1) * 100,
    "lustre_write_change_pct": (cur_lus_w / prior["lustre_write_gbps"] - 1) * 100,
    "lustre_read_change_pct": (cur_lus_r / prior["lustre_cold_gbps"] - 1) * 100,
}

# New features (not in prior version)
print()
print(f"  {'NEW: LRU Eviction Write':<30} {results['tests']['lru_eviction']['write_gbps']:.2f} GB/s")
print(f"  {'NEW: LRU Eviction Read':<30} {results['tests']['lru_eviction']['read_gbps']:.2f} GB/s")
print(f"  {'NEW: Integrated Write':<30} {results['tests']['integrated_store']['write_gbps']:.2f} GB/s")
print(f"  {'NEW: Integrated Read':<30} {results['tests']['integrated_store']['read_gbps']:.2f} GB/s")
print(f"  {'NEW: Dedup Write':<30} {results['tests']['integrated_store']['dedup_write_gbps']:.2f} GB/s")
print(f"  {'NEW: Semantic Eviction':<30} {'PASS ✓' if results['tests']['semantic_eviction']['pass'] else 'FAIL ✗'}")

# Save results
with open(RESULT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {RESULT_FILE}")

print()
print("═" * 60)
print("  Benchmark Complete!")
print("═" * 60)
PYEOF

echo
echo "Job completed at $(date)"
echo "Results: $RESULT_FILE"
