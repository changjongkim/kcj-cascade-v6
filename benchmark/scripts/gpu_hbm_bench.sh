#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -J gpu_hbm
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/gpu_hbm_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/gpu_hbm_%j.err

set -e

echo "=================================================================="
echo "GPU HBM Bandwidth Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

module load cudatoolkit pytorch
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
print("GPU HBM & PCIe Bandwidth Benchmark")
print("="*80)

device = torch.device("cuda:0")
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")

sizes_mb = [64, 256, 512, 1024]
results = {}

print("\n" + "-"*80)
print(f"{'Size':<15} {'D2D (HBM)':<25} {'H2D (Write)':<25} {'D2H (Read)':<20}")
print("-"*80)

for size_mb in sizes_mb:
    size_bytes = size_mb * 1024 * 1024
    
    gpu_src = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    gpu_dst = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    cpu_pinned = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    
    gpu_src.random_(0, 256)
    cpu_pinned.random_(0, 256)
    
    # Warmup
    for _ in range(5):
        gpu_dst.copy_(gpu_src)
    torch.cuda.synchronize()
    
    # D2D (HBM bandwidth)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        gpu_dst.copy_(gpu_src)
    end.record()
    torch.cuda.synchronize()
    d2d_gbps = (size_bytes * 20 / 1e9) / (start.elapsed_time(end) / 1000)
    
    # H2D
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        gpu_dst.copy_(cpu_pinned)
    torch.cuda.synchronize()
    h2d_gbps = (size_bytes * 20 / 1e9) / (time.perf_counter() - t0)
    
    # D2H
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        cpu_pinned.copy_(gpu_src)
    torch.cuda.synchronize()
    d2h_gbps = (size_bytes * 20 / 1e9) / (time.perf_counter() - t0)
    
    results[f"{size_mb}MB"] = {"D2D": d2d_gbps, "H2D": h2d_gbps, "D2H": d2h_gbps}
    print(f"{size_mb:>6} MB       {d2d_gbps:>8.1f} GB/s         {h2d_gbps:>8.1f} GB/s         {d2h_gbps:>8.1f} GB/s")
    del gpu_src, gpu_dst, cpu_pinned
    torch.cuda.empty_cache()

# Summary
print("\n" + "="*80)
print("4-Tier Storage Hierarchy Bandwidth")
print("="*80)

best = results.get("512MB", results[list(results.keys())[-1]])
hbm = best["D2D"]
h2d = best["H2D"]
d2h = best["D2H"]

hbm_bar = "#" * min(int(hbm/20), 50)
h2d_bar = "#" * min(int(h2d/2), 50)
d2h_bar = "#" * min(int(d2h/2), 50)
shm_bar = "#" * min(int(160/2), 50)

print(f"""
+-----------------------------------------------------------------------------+
|                          Storage Bandwidth Hierarchy                         |
+-----------------------------------------------------------------------------+
|                                                                             |
| GPU HBM (Hot Data) - D2D cudaMemcpy                                        |
|     {hbm_bar:<55} {hbm:>8.1f} GB/s |
|                                                                             |
| PCIe H2D (CPU->GPU Write)                                                   |
|     {h2d_bar:<55} {h2d:>8.1f} GB/s |
|                                                                             |
| PCIe D2H (GPU->CPU Read)                                                    |
|     {d2h_bar:<55} {d2h:>8.1f} GB/s |
|                                                                             |
| Cascade SHM (C++ mmap)                                                      |
|     {shm_bar:<55} {"160.9":>8} GB/s |
|                                                                             |
| Lustre PFS                                                                  |
|     #                                                                  0.96 GB/s |
|                                                                             |
+-----------------------------------------------------------------------------+
""")

# Multi-GPU P2P
print("\n" + "="*80)
print("Multi-GPU P2P Bandwidth")  
print("="*80)

num_gpus = torch.cuda.device_count()
print(f"GPUs: {num_gpus}")

if num_gpus >= 2:
    size_bytes = 512 * 1024 * 1024
    for i in range(min(num_gpus, 4)):
        for j in range(min(num_gpus, 4)):
            if i != j:
                src = torch.empty(size_bytes, dtype=torch.uint8, device=f"cuda:{i}")
                dst = torch.empty(size_bytes, dtype=torch.uint8, device=f"cuda:{j}")
                src.random_(0, 256)
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    dst.copy_(src)
                torch.cuda.synchronize()
                gbps = (size_bytes * 10 / 1e9) / (time.perf_counter() - t0)
                
                can_p2p = torch.cuda.can_device_access_peer(i, j)
                link = "NVLink" if can_p2p else "PCIe"
                print(f"  GPU {i} -> GPU {j}: {gbps:>8.1f} GB/s ({link})")
                del src, dst

output_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/gpu_hbm_{job_id}.json"
with open(output_path, "w") as f:
    json.dump({
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "results": results
    }, f, indent=2)

print(f"\nResults saved: {output_path}")
PYTHON_END

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
