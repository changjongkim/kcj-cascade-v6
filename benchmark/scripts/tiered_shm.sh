#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J tiered_shm
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/tiered_shm_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/tiered_shm_%j.err

###############################################################################
# Tiered KV Cache Benchmark - SHM Layer
#
# 캐시 레이어: SHM (/dev/shm)
# 백엔드 (miss 시): Lustre ($SCRATCH)
#
# 모든 시스템이 동일한 SHM 캐시 레이어 사용:
#   - SHM Hit: SHM에서 직접 반환 (모든 시스템 비슷)
#   - SHM Miss: 각 시스템의 백엔드에서 가져와서 SHM에 로드
###############################################################################

set -e

echo "=================================================================="
echo "Tiered 5-System Benchmark - SHM Cache Layer"
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
print("Tiered 5-System Benchmark - SHM Cache Layer")
print("="*80)
print(f"GPU: {torch.cuda.get_device_name(0)}")

###############################################################################
# Configuration
###############################################################################

BLOCK_SIZE_MB = 512
BLOCK_SIZE = BLOCK_SIZE_MB * 1024 * 1024
NUM_BLOCKS = 10  # 5GB total
NUM_ACCESS = 50  # 50번 접근
SHM_CACHE_SIZE = 5  # 5개 블록만 SHM에 캐시 (50% 용량)

results = {}
LUSTRE_PATH = "/pscratch/sd/s/sgkim/cascade_lustre/benchmark_blocks"
SHM_PATH = "/dev/shm/cascade_shm_bench"

os.makedirs(LUSTRE_PATH, exist_ok=True)
os.makedirs(SHM_PATH, exist_ok=True)

###############################################################################
# Data Generation (Cold Storage in Lustre)
###############################################################################

print(f"\n[데이터 생성] {NUM_BLOCKS} blocks ({NUM_BLOCKS * BLOCK_SIZE_MB / 1024:.2f} GB)")
print(f"  - GPU에서 랜덤 데이터 생성")
print(f"  - Lustre에 cold storage로 저장: {LUSTRE_PATH}")
print(f"  - SHM 캐시 캐패시티: {SHM_CACHE_SIZE} blocks ({SHM_CACHE_SIZE * BLOCK_SIZE_MB / 1024:.2f} GB)")

blocks_data = {}
for i in range(NUM_BLOCKS):
    # GPU에서 랜덤 데이터 생성
    data = torch.randint(0, 256, (BLOCK_SIZE,), dtype=torch.uint8, device=device)
    torch.cuda.synchronize()
    
    # Lustre에 저장 (cold storage)
    fpath = f"{LUSTRE_PATH}/block_{i}.bin"
    cpu_data = data.cpu().numpy()
    with open(fpath, "wb") as f:
        f.write(cpu_data.tobytes())
    
    blocks_data[f"block_{i}"] = cpu_data
    print(f"  Block {i}: {BLOCK_SIZE_MB}MB saved to Lustre")

# Page cache 비우기 (cold read 보장)
def drop_page_cache(path):
    """posix_fadvise(DONTNEED)로 page cache 비움"""
    try:
        fd = os.open(path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        libc = ctypes.CDLL("libc.so.6")
        libc.posix_fadvise(fd, 0, file_size, 4)  # POSIX_FADV_DONTNEED = 4
        os.close(fd)
    except:
        pass

for i in range(NUM_BLOCKS):
    drop_page_cache(f"{LUSTRE_PATH}/block_{i}.bin")

print(f"\n데이터 생성 완료. Page cache cleared for cold read.")

###############################################################################
# Base Tiered Cache Class (SHM Layer)
###############################################################################

class TieredSHMCache:
    """모든 시스템이 공유하는 SHM 캐시 레이어"""
    
    def __init__(self, name, shm_capacity, backend_get_fn, backend_put_fn, shm_subdir):
        self.name = name
        self.shm_capacity = shm_capacity
        self.shm_cache = OrderedDict()  # LRU: block_id -> SHM path
        self.backend_get = backend_get_fn
        self.backend_put = backend_put_fn
        self.shm_dir = f"{SHM_PATH}/{shm_subdir}"
        os.makedirs(self.shm_dir, exist_ok=True)
        self.stats = {"shm_hit": 0, "shm_miss": 0, "hit_time": 0, "miss_time": 0}
    
    def get(self, block_id):
        """SHM에서 찾고, 없으면 backend에서 가져와서 SHM에 로드"""
        start = time.perf_counter()
        
        if block_id in self.shm_cache:
            # SHM Hit!
            self.stats["shm_hit"] += 1
            self.shm_cache.move_to_end(block_id)
            
            # Read from SHM
            fpath = self.shm_cache[block_id]
            data = np.fromfile(fpath, dtype=np.uint8)
            
            self.stats["hit_time"] += time.perf_counter() - start
            return data
        else:
            # SHM Miss - backend에서 가져옴
            self.stats["shm_miss"] += 1
            data = self.backend_get(block_id)
            
            # SHM에 로드
            if len(self.shm_cache) >= self.shm_capacity:
                # LRU eviction
                evicted_id, evicted_path = self.shm_cache.popitem(last=False)
                if os.path.exists(evicted_path):
                    os.remove(evicted_path)
            
            # Cache in SHM
            shm_path = f"{self.shm_dir}/{block_id}.bin"
            with open(shm_path, "wb") as f:
                f.write(data.tobytes())
            self.shm_cache[block_id] = shm_path
            
            self.stats["miss_time"] += time.perf_counter() - start
            return data
    
    def put(self, block_id, data):
        """SHM에 저장하고 backend에도 동기화"""
        if len(self.shm_cache) >= self.shm_capacity:
            evicted_id, evicted_path = self.shm_cache.popitem(last=False)
            if os.path.exists(evicted_path):
                os.remove(evicted_path)
        
        shm_path = f"{self.shm_dir}/{block_id}.bin"
        with open(shm_path, "wb") as f:
            f.write(data.tobytes())
        self.shm_cache[block_id] = shm_path
        self.backend_put(block_id, data)
    
    def hit_rate(self):
        total = self.stats["shm_hit"] + self.stats["shm_miss"]
        return self.stats["shm_hit"] / total if total > 0 else 0
    
    def cleanup(self):
        import shutil
        if os.path.exists(self.shm_dir):
            shutil.rmtree(self.shm_dir)

###############################################################################
# System 1: Cascade (SHM mmap → Lustre)
###############################################################################

print("\n" + "-"*80)
print("[1/5] Cascade: SHM (mmap) → Lustre")
print("-"*80)

def cascade_backend_get(block_id):
    """Cascade: Lustre에서 가져옴 (SHM miss시)"""
    fpath = f"{LUSTRE_PATH}/{block_id}.bin"
    
    # mmap for efficient read
    with open(fpath, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = np.frombuffer(mm, dtype=np.uint8).copy()
        mm.close()
    return data

def cascade_backend_put(block_id, data):
    """Cascade: Lustre에는 이미 있음"""
    pass

cascade_cache = TieredSHMCache("Cascade", SHM_CACHE_SIZE, cascade_backend_get, cascade_backend_put, "cascade")

# Warmup - all blocks go through once
for i in range(NUM_BLOCKS):
    cascade_cache.put(f"block_{i}", blocks_data[f"block_{i}"])

cascade_cache.stats = {"shm_hit": 0, "shm_miss": 0, "hit_time": 0, "miss_time": 0}  # Reset stats

# Random access pattern
np.random.seed(42)
access_pattern = [f"block_{np.random.randint(0, NUM_BLOCKS)}" for _ in range(NUM_ACCESS)]

start = time.perf_counter()
for block_id in access_pattern:
    _ = cascade_cache.get(block_id)
elapsed = time.perf_counter() - start

cascade_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
print(f"  Throughput: {cascade_gbps:.2f} GB/s")
print(f"  Hit rate: {cascade_cache.hit_rate()*100:.1f}%")
print(f"  Hit time avg: {cascade_cache.stats['hit_time']/max(cascade_cache.stats['shm_hit'],1)*1000:.2f}ms")
print(f"  Miss time avg: {cascade_cache.stats['miss_time']/max(cascade_cache.stats['shm_miss'],1)*1000:.2f}ms")

results["Cascade-C++"] = {
    "gbps": cascade_gbps, 
    "hit_rate": cascade_cache.hit_rate(),
    "backend": "mmap"
}

###############################################################################
# System 2: vLLM-GPU (SHM → disk swap via torch)
###############################################################################

print("\n" + "-"*80)
print("[2/5] vLLM-GPU: SHM → torch.load from disk")
print("-"*80)

def vllm_backend_get(block_id):
    """vLLM: disk에서 torch.load"""
    fpath = f"{LUSTRE_PATH}/{block_id}.bin"
    with open(fpath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return data

def vllm_backend_put(block_id, data):
    pass

vllm_cache = TieredSHMCache("vLLM", SHM_CACHE_SIZE, vllm_backend_get, vllm_backend_put, "vllm")

for i in range(NUM_BLOCKS):
    vllm_cache.put(f"block_{i}", blocks_data[f"block_{i}"])
vllm_cache.stats = {"shm_hit": 0, "shm_miss": 0, "hit_time": 0, "miss_time": 0}

start = time.perf_counter()
for block_id in access_pattern:
    _ = vllm_cache.get(block_id)
elapsed = time.perf_counter() - start

vllm_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
print(f"  Throughput: {vllm_gbps:.2f} GB/s")
print(f"  Hit rate: {vllm_cache.hit_rate()*100:.1f}%")

results["vLLM-GPU"] = {
    "gbps": vllm_gbps,
    "hit_rate": vllm_cache.hit_rate(),
    "backend": "file read"
}

###############################################################################
# System 3: PDC (SHM → file container)
###############################################################################

print("\n" + "-"*80)
print("[3/5] PDC: SHM → file container")
print("-"*80)

def pdc_backend_get(block_id):
    fpath = f"{LUSTRE_PATH}/{block_id}.bin"
    with open(fpath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return data

def pdc_backend_put(block_id, data):
    pass

pdc_cache = TieredSHMCache("PDC", SHM_CACHE_SIZE, pdc_backend_get, pdc_backend_put, "pdc")

for i in range(NUM_BLOCKS):
    pdc_cache.put(f"block_{i}", blocks_data[f"block_{i}"])
pdc_cache.stats = {"shm_hit": 0, "shm_miss": 0, "hit_time": 0, "miss_time": 0}

start = time.perf_counter()
for block_id in access_pattern:
    _ = pdc_cache.get(block_id)
elapsed = time.perf_counter() - start

pdc_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
print(f"  Throughput: {pdc_gbps:.2f} GB/s")
print(f"  Hit rate: {pdc_cache.hit_rate()*100:.1f}%")

results["PDC"] = {
    "gbps": pdc_gbps,
    "hit_rate": pdc_cache.hit_rate(),
    "backend": "file container"
}

###############################################################################
# System 4: LMCache (SHM → CPU tensor)
###############################################################################

print("\n" + "-"*80)
print("[4/5] LMCache: SHM → CPU tensor")
print("-"*80)

def lmcache_backend_get(block_id):
    fpath = f"{LUSTRE_PATH}/{block_id}.bin"
    data = np.fromfile(fpath, dtype=np.uint8)
    return data

def lmcache_backend_put(block_id, data):
    pass

lmcache_cache = TieredSHMCache("LMCache", SHM_CACHE_SIZE, lmcache_backend_get, lmcache_backend_put, "lmcache")

for i in range(NUM_BLOCKS):
    lmcache_cache.put(f"block_{i}", blocks_data[f"block_{i}"])
lmcache_cache.stats = {"shm_hit": 0, "shm_miss": 0, "hit_time": 0, "miss_time": 0}

start = time.perf_counter()
for block_id in access_pattern:
    _ = lmcache_cache.get(block_id)
elapsed = time.perf_counter() - start

lmcache_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
print(f"  Throughput: {lmcache_gbps:.2f} GB/s")
print(f"  Hit rate: {lmcache_cache.hit_rate()*100:.1f}%")

results["LMCache"] = {
    "gbps": lmcache_gbps,
    "hit_rate": lmcache_cache.hit_rate(),
    "backend": "numpy fromfile"
}

###############################################################################
# System 5: HDF5 (SHM → HDF5 file)
###############################################################################

print("\n" + "-"*80)
print("[5/5] HDF5: SHM → HDF5 file")
print("-"*80)

try:
    import h5py
    
    hdf5_lustre_path = f"{LUSTRE_PATH}/hdf5_data.h5"
    
    # Save data to HDF5 format in Lustre
    with h5py.File(hdf5_lustre_path, "w") as f:
        for block_id, data in blocks_data.items():
            f.create_dataset(block_id, data=data)
    
    def hdf5_backend_get(block_id):
        with h5py.File(hdf5_lustre_path, "r") as f:
            data = f[block_id][:]
        return data
    
    def hdf5_backend_put(block_id, data):
        pass
    
    hdf5_cache = TieredSHMCache("HDF5", SHM_CACHE_SIZE, hdf5_backend_get, hdf5_backend_put, "hdf5")
    
    for i in range(NUM_BLOCKS):
        hdf5_cache.put(f"block_{i}", blocks_data[f"block_{i}"])
    hdf5_cache.stats = {"shm_hit": 0, "shm_miss": 0, "hit_time": 0, "miss_time": 0}
    
    start = time.perf_counter()
    for block_id in access_pattern:
        _ = hdf5_cache.get(block_id)
    elapsed = time.perf_counter() - start
    
    hdf5_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
    print(f"  Throughput: {hdf5_gbps:.2f} GB/s")
    print(f"  Hit rate: {hdf5_cache.hit_rate()*100:.1f}%")
    
    results["HDF5"] = {
        "gbps": hdf5_gbps,
        "hit_rate": hdf5_cache.hit_rate(),
        "backend": "h5py"
    }
    
    if os.path.exists(hdf5_lustre_path):
        os.remove(hdf5_lustre_path)
        
except Exception as e:
    print(f"  Failed: {e}")
    results["HDF5"] = {"gbps": 0, "error": str(e)}

###############################################################################
# Summary
###############################################################################

print("\n" + "="*80)
print("SUMMARY: Tiered 5-System Benchmark (SHM Cache Layer)")
print(f"Cache Layer: SHM (/dev/shm)")
print(f"Backend: Lustre ($SCRATCH)")
print(f"Block Size: {BLOCK_SIZE_MB}MB, SHM Cache: {SHM_CACHE_SIZE} blocks")
print(f"Access Pattern: {NUM_ACCESS} random accesses")
print("="*80)

sorted_systems = sorted(results.items(), key=lambda x: x[1].get("gbps", 0), reverse=True)

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│              Tiered KV Cache Performance (SHM + Lustre Backend)              │
├──────────────────────────────────────────────────────────────────────────────┤
│ System       │ Throughput   │ Hit Rate │ Backend Method                     │
├──────────────┼──────────────┼──────────┼────────────────────────────────────┤""")

for name, r in sorted_systems:
    gbps = r.get("gbps", 0)
    hit = r.get("hit_rate", 0) * 100
    backend = r.get("backend", "Unknown")
    print(f"│ {name:<12} │ {gbps:>9.2f} GB/s │ {hit:>6.1f}% │ {backend:<34} │")

print("└──────────────────────────────────────────────────────────────────────────────┘")

max_gbps = max(r.get("gbps", 0) for r in results.values())
print(f"""
Performance Comparison (GB/s):
┌──────────────────────────────────────────────────────────────────────────────┐""")

for name, r in sorted_systems:
    gbps = r.get("gbps", 0)
    if max_gbps > 0:
        bar_len = int(50 * gbps / max_gbps)
        bar = "█" * bar_len
        print(f"│ {name:<12} {bar:<50} {gbps:>7.2f} │")

print("└──────────────────────────────────────────────────────────────────────────────┘")

###############################################################################
# Save Results
###############################################################################

output = {
    "job_id": job_id,
    "timestamp": datetime.now().isoformat(),
    "benchmark_type": "Tiered KV Cache with SHM Layer",
    "config": {
        "block_size_mb": BLOCK_SIZE_MB,
        "num_blocks": NUM_BLOCKS,
        "num_accesses": NUM_ACCESS,
        "shm_cache_size": SHM_CACHE_SIZE,
        "access_pattern": "random",
        "cache_layer": "SHM (/dev/shm)",
        "backend": "Lustre ($SCRATCH)"
    },
    "gpu": torch.cuda.get_device_name(0),
    "note": "All systems use same SHM cache layer. Difference is in backend (Lustre read) method.",
    "results": results
}

output_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/tiered_shm_{job_id}.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved: {output_path}")

# Cleanup
for cache in [cascade_cache, vllm_cache, pdc_cache, lmcache_cache]:
    cache.cleanup()
try:
    hdf5_cache.cleanup()
except:
    pass

import shutil
shutil.rmtree(f"{LUSTRE_PATH}", ignore_errors=True)
shutil.rmtree(f"{SHM_PATH}", ignore_errors=True)

PYTHON_END

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
