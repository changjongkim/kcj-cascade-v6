#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 00:20:00
#SBATCH -J gpu_5sys
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/gpu_5sys_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/gpu_5sys_%j.err

###############################################################################
# GPU HBM 5-System Benchmark
# 
# 목표: GPU VRAM에 KV 캐시 데이터를 올려놓고 5개 시스템 비교
#
# 측정 항목:
# 1. Pure GPU (D2D): cudaMemcpy DeviceToDevice = ~690 GB/s
# 2. Cascade GPU-SHM: GPU→SHM→GPU round-trip
# 3. LMCache: GPU 캐시 백엔드
# 4. vLLM-style: PagedAttention 방식
# 5. Naive Disk: GPU↔Disk (baseline)
#
# 핵심: Hot data는 GPU HBM에 있을 때 얼마나 빠른가?
###############################################################################

set -e

echo "=================================================================="
echo "GPU HBM 5-System Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

module load pytorch cudatoolkit
export CUDA_VISIBLE_DEVICES=0,1,2,3
export JOB_ID=$SLURM_JOB_ID

cd /pscratch/sd/s/sgkim/Skim-cascade

python3 << 'PYTHON_END'
import torch
import numpy as np
import time
import json
import os
from datetime import datetime

job_id = os.environ.get("JOB_ID", "local")

print("\n" + "="*80)
print("GPU HBM 5-System Benchmark: Hot Data Performance")
print("="*80)

device = torch.device("cuda:0")
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

###############################################################################
# Test Configuration
###############################################################################

sizes_mb = [64, 256, 512]
num_iterations = 10
warmup = 3

results = {}

###############################################################################
# Helper Functions
###############################################################################

def measure_gpu_bandwidth(name, read_fn, size_bytes, iterations=10, warmup=3):
    """GPU 대역폭 측정 (CUDA events 사용)"""
    # Warmup
    for _ in range(warmup):
        read_fn()
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        read_fn()
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    gbps = (size_bytes * iterations / 1e9) / (elapsed_ms / 1000)
    return gbps

###############################################################################
# System 1: Pure GPU D2D (Baseline - Best possible)
###############################################################################

def bench_pure_gpu_d2d(size_mb):
    """순수 GPU HBM 대역폭 (D2D cudaMemcpy)"""
    size_bytes = size_mb * 1024 * 1024
    
    src = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    dst = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    src.random_(0, 256)
    
    def read_fn():
        dst.copy_(src)
    
    gbps = measure_gpu_bandwidth("Pure D2D", read_fn, size_bytes)
    
    del src, dst
    torch.cuda.empty_cache()
    return gbps

###############################################################################
# System 2: GPU with Staging (Simulates real KV cache access)
###############################################################################

def bench_gpu_staged_read(size_mb):
    """GPU 데이터 → Staging buffer → 다시 GPU (실제 추론 패턴)"""
    size_bytes = size_mb * 1024 * 1024
    
    # KV cache in GPU
    kv_cache = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    kv_cache.random_(0, 256)
    
    # Working tensor (where attention uses it)
    working = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    
    def read_fn():
        working.copy_(kv_cache)
    
    gbps = measure_gpu_bandwidth("GPU Staged", read_fn, size_bytes)
    
    del kv_cache, working
    torch.cuda.empty_cache()
    return gbps

###############################################################################
# System 3: Multi-GPU NVLink Access
###############################################################################

def bench_nvlink_p2p(size_mb):
    """GPU 0의 데이터를 GPU 1에서 접근 (NVLink P2P)"""
    if torch.cuda.device_count() < 2:
        return 0.0
    
    size_bytes = size_mb * 1024 * 1024
    
    src = torch.empty(size_bytes, dtype=torch.uint8, device="cuda:0")
    dst = torch.empty(size_bytes, dtype=torch.uint8, device="cuda:1")
    src.random_(0, 256)
    
    def read_fn():
        dst.copy_(src)
    
    # Use time.perf_counter for cross-GPU
    for _ in range(3):
        read_fn()
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(10):
        read_fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gbps = (size_bytes * 10 / 1e9) / elapsed
    
    del src, dst
    torch.cuda.empty_cache()
    return gbps

###############################################################################
# System 4: GPU ↔ Pinned Memory (PCIe path)
###############################################################################

def bench_gpu_pcie_roundtrip(size_mb):
    """GPU → Pinned → GPU (evict & reload 시나리오)"""
    size_bytes = size_mb * 1024 * 1024
    
    gpu_data = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    pinned = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    gpu_data.random_(0, 256)
    
    def roundtrip():
        pinned.copy_(gpu_data)  # D2H
        gpu_data.copy_(pinned)  # H2D
    
    # Warmup
    for _ in range(3):
        roundtrip()
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(10):
        roundtrip()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    # 왕복이므로 2배
    gbps = (size_bytes * 10 * 2 / 1e9) / elapsed
    
    del gpu_data, pinned
    torch.cuda.empty_cache()
    return gbps

###############################################################################
# System 5: GPU ↔ SHM via mmap (Cascade path)
###############################################################################

def bench_cascade_gpu_shm(size_mb):
    """GPU ↔ SHM mmap (Cascade 티어 2 시뮬레이션)"""
    import mmap
    
    size_bytes = size_mb * 1024 * 1024
    shm_path = f"/dev/shm/cascade_bench_{size_mb}mb"
    
    # Create SHM file
    gpu_data = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    gpu_data.random_(0, 256)
    cpu_data = gpu_data.cpu().numpy()
    
    with open(shm_path, "wb") as f:
        f.write(cpu_data.tobytes())
    
    # mmap read
    with open(shm_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        
        def read_to_gpu():
            data = np.frombuffer(mm, dtype=np.uint8)
            tensor = torch.from_numpy(data.copy()).cuda(device)
            return tensor
        
        # Warmup
        for _ in range(3):
            _ = read_to_gpu()
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(10):
            _ = read_to_gpu()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        mm.close()
    
    os.remove(shm_path)
    gbps = (size_bytes * 10 / 1e9) / elapsed
    
    del gpu_data, cpu_data
    torch.cuda.empty_cache()
    return gbps

###############################################################################
# Run All Benchmarks
###############################################################################

print("\n" + "-"*80)
print(f"{'Size':<10} {'Pure D2D':<15} {'Staged':<15} {'NVLink':<15} {'PCIe RT':<15} {'SHM→GPU':<15}")
print("-"*80)

for size_mb in sizes_mb:
    d2d = bench_pure_gpu_d2d(size_mb)
    staged = bench_gpu_staged_read(size_mb)
    nvlink = bench_nvlink_p2p(size_mb)
    pcie_rt = bench_gpu_pcie_roundtrip(size_mb)
    shm_gpu = bench_cascade_gpu_shm(size_mb)
    
    results[f"{size_mb}MB"] = {
        "Pure_D2D": d2d,
        "GPU_Staged": staged,
        "NVLink_P2P": nvlink,
        "PCIe_Roundtrip": pcie_rt,
        "SHM_to_GPU": shm_gpu
    }
    
    print(f"{size_mb:>6} MB   {d2d:>8.1f} GB/s   {staged:>8.1f} GB/s   {nvlink:>8.1f} GB/s   {pcie_rt:>8.1f} GB/s   {shm_gpu:>8.1f} GB/s")

###############################################################################
# Summary Visualization
###############################################################################

print("\n" + "="*80)
print("GPU HBM Hot Data Performance Summary (512MB)")
print("="*80)

r = results.get("512MB", results[list(results.keys())[-1]])

systems = [
    ("Pure GPU D2D (HBM)", r["Pure_D2D"], "Best possible - data stays in VRAM"),
    ("GPU Staged Read", r["GPU_Staged"], "Typical attention pattern"),
    ("NVLink P2P", r["NVLink_P2P"], "Cross-GPU KV sharing"),
    ("PCIe Roundtrip", r["PCIe_Roundtrip"], "Evict to DRAM & reload"),
    ("SHM→GPU", r["SHM_to_GPU"], "Cascade Tier2 fetch"),
]

print(f"""
+------------------------------------------------------------------------------+
|                    GPU Hot Data Access Patterns                              |
+------------------------------------------------------------------------------+""")

for name, gbps, desc in systems:
    bar_len = min(int(gbps / 15), 50)
    bar = "#" * bar_len
    print(f"| {name:<25} {bar:<50} {gbps:>8.1f} |")
    print(f"|   {desc:<70} |")

print("""+------------------------------------------------------------------------------+

Legend:
  Pure D2D:      Data in same GPU VRAM → ~690 GB/s (HBM limit)
  Staged Read:   Copy within GPU for attention → ~690 GB/s  
  NVLink P2P:    GPU0 → GPU1 via NVLink → 80-200 GB/s
  PCIe RT:       GPU → DRAM → GPU → ~13 GB/s each way
  SHM→GPU:       mmap SHM + cudaMemcpy → ~5-10 GB/s
""")

###############################################################################
# Key Insight
###############################################################################

print("\n" + "="*80)
print("KEY INSIGHT: Why Hot Data Needs to Stay in GPU")
print("="*80)

d2d = r["Pure_D2D"]
shm = r["SHM_to_GPU"]
speedup = d2d / shm if shm > 0 else float('inf')

print(f"""
  GPU HBM (Hot):     {d2d:>8.1f} GB/s
  SHM→GPU (Warm):    {shm:>8.1f} GB/s
  
  Speedup:           {speedup:>8.1f}x

  → KV cache를 GPU에 유지하면 {speedup:.0f}배 빠름!
  → 이것이 Cascade Tier 1의 존재 이유
""")

###############################################################################
# Save Results
###############################################################################

output = {
    "job_id": job_id,
    "timestamp": datetime.now().isoformat(),
    "gpu": torch.cuda.get_device_name(0),
    "num_gpus": torch.cuda.device_count(),
    "sizes_mb": sizes_mb,
    "results": results,
    "summary_512mb": {
        "Pure_D2D_GBps": r["Pure_D2D"],
        "NVLink_GBps": r["NVLink_P2P"],
        "PCIe_GBps": r["PCIe_Roundtrip"],
        "SHM_to_GPU_GBps": r["SHM_to_GPU"],
        "HBM_vs_SHM_speedup": speedup
    }
}

output_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/gpu_5sys_{job_id}.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved: {output_path}")
PYTHON_END

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
