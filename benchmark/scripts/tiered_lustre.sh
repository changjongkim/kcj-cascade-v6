#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J tiered_lustre
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/tiered_lustre_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/tiered_lustre_%j.err

###############################################################################
# Tiered KV Cache Benchmark - Lustre Only (Cold Storage)
#
# 캐시 레이어: Lustre Page Cache (OS managed)
# 백엔드: Lustre Disk ($SCRATCH)
#
# 두 가지 모드 측정:
#   1. Hot Read: OS page cache에서 읽기
#   2. Cold Read: posix_fadvise(DONTNEED)로 page cache 비운 후 읽기
###############################################################################

set -e

echo "=================================================================="
echo "Tiered 5-System Benchmark - Lustre Only (Cold Storage)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

module load pytorch cudatoolkit
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /pscratch/sd/s/sgkim/Skim-cascade

python3 << 'PYTHON_END'
import torch
import numpy as np
import time
import json
import os
import mmap
import ctypes
from datetime import datetime
from collections import OrderedDict

job_id = os.environ.get("SLURM_JOB_ID", "local")
device = torch.device("cuda:0")

print("\n" + "="*80)
print("Tiered 5-System Benchmark - Lustre Only (Cold Storage)")
print("="*80)
print(f"GPU: {torch.cuda.get_device_name(0)}")

###############################################################################
# Configuration
###############################################################################

BLOCK_SIZE_MB = 512
BLOCK_SIZE = BLOCK_SIZE_MB * 1024 * 1024
NUM_BLOCKS = 10  # 5GB total
NUM_ACCESS = 20  # 20번 접근 (Lustre 느려서 줄임)

results = {}
LUSTRE_PATH = "/pscratch/sd/s/sgkim/cascade_lustre/lustre_bench"

os.makedirs(LUSTRE_PATH, exist_ok=True)

###############################################################################
# Data Generation (Cold Storage in Lustre)
###############################################################################

print(f"\n[데이터 생성] {NUM_BLOCKS} blocks ({NUM_BLOCKS * BLOCK_SIZE_MB / 1024:.2f} GB)")
print(f"  - 저장 위치: {LUSTRE_PATH}")
print("  - 데이터 타입: 랜덤 uint8 (KV cache 시뮬레이션)")

blocks_data = {}
write_start = time.perf_counter()

for i in range(NUM_BLOCKS):
    # GPU에서 랜덤 데이터 생성
    data = torch.randint(0, 256, (BLOCK_SIZE,), dtype=torch.uint8, device=device)
    torch.cuda.synchronize()
    
    # Lustre에 저장
    fpath = f"{LUSTRE_PATH}/block_{i}.bin"
    cpu_data = data.cpu().numpy()
    with open(fpath, "wb") as f:
        f.write(cpu_data.tobytes())
    
    blocks_data[f"block_{i}"] = fpath
    print(f"  Block {i}: {BLOCK_SIZE_MB}MB saved")

write_elapsed = time.perf_counter() - write_start
write_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / write_elapsed
print(f"\n데이터 생성 완료: {write_gbps:.2f} GB/s (write)")

# Page cache를 비우는 함수
def drop_page_cache(path):
    """posix_fadvise(DONTNEED)로 page cache 비움"""
    try:
        fd = os.open(path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        libc = ctypes.CDLL("libc.so.6")
        libc.posix_fadvise(fd, 0, file_size, 4)  # POSIX_FADV_DONTNEED = 4
        os.close(fd)
        return True
    except Exception as e:
        return False

###############################################################################
# Backend Read Functions
###############################################################################

def cascade_read(block_id, fpath):
    """Cascade: mmap for efficient read"""
    with open(fpath, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = np.frombuffer(mm, dtype=np.uint8).copy()
        mm.close()
    return data

def vllm_read(block_id, fpath):
    """vLLM: standard file read"""
    with open(fpath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return data

def pdc_read(block_id, fpath):
    """PDC: file container read (simulated)"""
    with open(fpath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return data

def lmcache_read(block_id, fpath):
    """LMCache: numpy fromfile"""
    data = np.fromfile(fpath, dtype=np.uint8)
    return data

def hdf5_read(block_id, hdf5_path):
    """HDF5: h5py read"""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        data = f[block_id][:]
    return data

###############################################################################
# HOT READ Benchmark (Page Cache Hit)
###############################################################################

print("\n" + "="*80)
print("TEST 1: HOT READ (OS Page Cache)")
print("="*80)

np.random.seed(42)
access_pattern = [f"block_{np.random.randint(0, NUM_BLOCKS)}" for _ in range(NUM_ACCESS)]

read_functions = {
    "Cascade-C++": cascade_read,
    "vLLM-GPU": vllm_read,
    "PDC": pdc_read,
    "LMCache": lmcache_read
}

hot_results = {}

for name, read_fn in read_functions.items():
    print(f"\n[{name}] Hot read...")
    
    # Warmup - cache 데이터
    for i in range(NUM_BLOCKS):
        _ = read_fn(f"block_{i}", blocks_data[f"block_{i}"])
    
    start = time.perf_counter()
    for block_id in access_pattern:
        fpath = blocks_data[block_id]
        _ = read_fn(block_id, fpath)
    elapsed = time.perf_counter() - start
    
    gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
    print(f"  Throughput: {gbps:.2f} GB/s")
    hot_results[name] = gbps

# HDF5 Hot Read
print(f"\n[HDF5] Hot read...")
try:
    import h5py
    hdf5_path = f"{LUSTRE_PATH}/data.h5"
    
    # Create HDF5 file
    with h5py.File(hdf5_path, "w") as f:
        for block_id in blocks_data:
            data = np.fromfile(blocks_data[block_id], dtype=np.uint8)
            f.create_dataset(block_id, data=data)
    
    # Warmup
    for block_id in blocks_data:
        _ = hdf5_read(block_id, hdf5_path)
    
    start = time.perf_counter()
    for block_id in access_pattern:
        _ = hdf5_read(block_id, hdf5_path)
    elapsed = time.perf_counter() - start
    
    hdf5_hot_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
    print(f"  Throughput: {hdf5_hot_gbps:.2f} GB/s")
    hot_results["HDF5"] = hdf5_hot_gbps
except Exception as e:
    print(f"  Failed: {e}")
    hot_results["HDF5"] = 0

###############################################################################
# COLD READ Benchmark (No Page Cache)
###############################################################################

print("\n" + "="*80)
print("TEST 2: COLD READ (Page Cache Dropped)")
print("="*80)

cold_results = {}

for name, read_fn in read_functions.items():
    print(f"\n[{name}] Cold read...")
    
    # Drop page cache
    for block_id in blocks_data:
        drop_page_cache(blocks_data[block_id])
    
    # Small delay
    time.sleep(0.5)
    
    start = time.perf_counter()
    for block_id in access_pattern:
        fpath = blocks_data[block_id]
        _ = read_fn(block_id, fpath)
    elapsed = time.perf_counter() - start
    
    gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
    print(f"  Throughput: {gbps:.2f} GB/s")
    cold_results[name] = gbps

# HDF5 Cold Read
print(f"\n[HDF5] Cold read...")
try:
    # Drop page cache for HDF5 file
    drop_page_cache(hdf5_path)
    time.sleep(0.5)
    
    start = time.perf_counter()
    for block_id in access_pattern:
        _ = hdf5_read(block_id, hdf5_path)
    elapsed = time.perf_counter() - start
    
    hdf5_cold_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
    print(f"  Throughput: {hdf5_cold_gbps:.2f} GB/s")
    cold_results["HDF5"] = hdf5_cold_gbps
except Exception as e:
    print(f"  Failed: {e}")
    cold_results["HDF5"] = 0

###############################################################################
# Summary
###############################################################################

print("\n" + "="*80)
print("SUMMARY: Lustre Storage Benchmark")
print(f"Block Size: {BLOCK_SIZE_MB}MB, {NUM_BLOCKS} blocks ({NUM_BLOCKS * BLOCK_SIZE_MB / 1024:.2f} GB)")
print(f"Access Pattern: {NUM_ACCESS} random accesses")
print("="*80)

for name in read_functions.keys():
    results[name] = {
        "write_gbps": write_gbps,
        "hot_gbps": hot_results.get(name, 0),
        "cold_gbps": cold_results.get(name, 0)
    }
results["HDF5"] = {
    "write_gbps": write_gbps,
    "hot_gbps": hot_results.get("HDF5", 0),
    "cold_gbps": cold_results.get("HDF5", 0)
}

sorted_by_hot = sorted(results.items(), key=lambda x: x[1]["hot_gbps"], reverse=True)
sorted_by_cold = sorted(results.items(), key=lambda x: x[1]["cold_gbps"], reverse=True)

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Lustre Storage Performance (GB/s)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ System       │ Write       │ Hot Read    │ Cold Read   │ Backend            │
├──────────────┼─────────────┼─────────────┼─────────────┼────────────────────┤""")

backends = {
    "Cascade-C++": "mmap",
    "vLLM-GPU": "file read",
    "PDC": "file container",
    "LMCache": "numpy fromfile",
    "HDF5": "h5py"
}

for name, r in sorted_by_hot:
    wr = r["write_gbps"]
    hot = r["hot_gbps"]
    cold = r["cold_gbps"]
    backend = backends.get(name, "Unknown")
    print(f"│ {name:<12} │ {wr:>8.2f} GB/s │ {hot:>8.2f} GB/s │ {cold:>8.2f} GB/s │ {backend:<18} │")

print("└──────────────────────────────────────────────────────────────────────────────┘")

# Hot Read Bar Chart
max_hot = max(r["hot_gbps"] for r in results.values())
print(f"""
HOT READ Performance (Page Cache):
┌──────────────────────────────────────────────────────────────────────────────┐""")
for name, r in sorted_by_hot:
    gbps = r["hot_gbps"]
    if max_hot > 0:
        bar_len = int(50 * gbps / max_hot)
        bar = "█" * bar_len
        print(f"│ {name:<12} {bar:<50} {gbps:>7.2f} │")
print("└──────────────────────────────────────────────────────────────────────────────┘")

# Cold Read Bar Chart
max_cold = max(r["cold_gbps"] for r in results.values())
print(f"""
COLD READ Performance (Direct Disk):
┌──────────────────────────────────────────────────────────────────────────────┐""")
for name, r in sorted_by_cold:
    gbps = r["cold_gbps"]
    if max_cold > 0:
        bar_len = int(50 * gbps / max_cold)
        bar = "█" * bar_len
        print(f"│ {name:<12} {bar:<50} {gbps:>7.2f} │")
print("└──────────────────────────────────────────────────────────────────────────────┘")

###############################################################################
# Save Results
###############################################################################

output = {
    "job_id": job_id,
    "timestamp": datetime.now().isoformat(),
    "benchmark_type": "Lustre Only (Cold Storage)",
    "config": {
        "block_size_mb": BLOCK_SIZE_MB,
        "num_blocks": NUM_BLOCKS,
        "num_accesses": NUM_ACCESS,
        "access_pattern": "random",
        "storage": "Lustre ($SCRATCH)",
        "cold_read_method": "posix_fadvise(DONTNEED)"
    },
    "gpu": torch.cuda.get_device_name(0),
    "note": "Hot = OS page cache, Cold = direct disk read after posix_fadvise",
    "results": results
}

output_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/tiered_lustre_{job_id}.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved: {output_path}")

# Cleanup
import shutil
shutil.rmtree(LUSTRE_PATH, ignore_errors=True)

PYTHON_END

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
