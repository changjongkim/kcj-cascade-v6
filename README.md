# 🚀 Cascade V6: Distributed 5-Tier KV Cache for HPC-Scale LLM Inference

<p align="center">
  <img src="https://img.shields.io/badge/SC'26-Target-blue?style=for-the-badge" alt="SC'26"/>
  <img src="https://img.shields.io/badge/NERSC-Perlmutter-green?style=for-the-badge" alt="Perlmutter"/>
  <img src="https://img.shields.io/badge/A100-SXM4-76B900?style=for-the-badge&logo=nvidia" alt="A100"/>
  <img src="https://img.shields.io/badge/Scale-8%20Nodes%20Verified-orange?style=for-the-badge" alt="Scale"/>
</p>

> **Core Metric:** Scaling from **1 Node (0.5 GB/s)** to **8 Nodes (3.9 GB/s)** Write Throughput with **96% linear efficiency**.
> **Performance:** Achieved **160 GB/s Aggregate Read Bandwidth** in 8-node Strong Scaling (Cache Hit).
> **Goal:** Solving the "Memory Capacity Wall" in LLM Inference by unifying local HBM, DRAM, Remote Memory, and Lustre PFS.

---

## 📖 Introduction: The Memory Wall in LLM Serving

As Large Language Models (LLMs) like Llama-3-70B scale to **128K+ context windows**, the Key-Value (KV) cache becomes the primary bottleneck, consuming hundreds of gigabytes per request. Single-node GPU memory (HBM) is insufficient, leading to:

1.  **Capacity Wall:** A 70B model with long context can only serve **<10 concurrent requests** on an A100 node.
2.  **Bandwidth Wall:** Evicting to disk (Lustre) is **1000x slower** than HBM, causing massive latency spikes during cache misses.
3.  **Redundancy:** In multi-tenant serving, identical "System Prompts" are duplicated across thousands of requests, wasting memory.

**Cascade V6** addresses these challenges via a **novel distributed hierarchy** that aggregates memory resources across HPC clusters.

---

## 🏗️ 5-Tier Memory Hierarchy Architecture

Cascade enables zero-copy access to hot data while providing near-infinite capacity for cold data.

| Tier | Resource | Bandwidth (Measured) | Latency | Capacity (Per Node) | Logical Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Local GPU HBM** (A100) | **1,555 GB/s** | ~0.5 μs | 160 GB (4x40GB) | **Hot Cache** (Active Tokens) |
| **Tier 2** | **Local DRAM** (Pinned) | **160+ GB/s** | ~10 μs | 256 GB | **Warm Cache** (Recent Context) |
| **Tier 3** | **Remote GPU** (NVLink/RDMA) | **22+ GB/s** | ~50 μs | N × 160 GB | **Neighbor Cache** (Elasticity) |
| **Tier 4** | **Remote DRAM** (RDMA) | **18 GB/s** | ~80 μs | N × 256 GB | **Cluster Pool** (Massive Capacity) |
| **Tier 5** | **Lustre PFS** ($SCRATCH) | **1~3 GB/s** | ~ms | 44 PB (Shared) | **Cold Storage** (Persistence) |

### Data Flow Diagram
```
[Inference Request]
       │
       ▼
   (Tier 1: GPU HBM) ── Hit? ──► [Zero-Copy Access]
       │ Miss
       ▼
   (Tier 2: Local DRAM) ── Hit? ──► [DMA Transfer 25GB/s]
       │ Miss
       ▼
   (Tier 3/4: Remote Pool) ── Hit? ──► [RDMA Transfer 22GB/s]
       │ Miss
       ▼
   (Tier 5: Lustre) ──► [Parallel IO Read]
```

---

## 🏆 Core Novelties (SC26 Contributions)

### 1. 🧠 Cross-Node Semantic Eviction (Novelty 1)
*   **The Problem:** Standard eviction policies (LRU) are content-agnostic. They evict "System Prompts" (critical for every request) just as easily as random tokens.
*   **Our Solution:** Cascade introduces **Semantic-Awareness**.
    *   **Prefix Blocks:** Identified and marked as "Protected".
    *   **Global Registry:** All nodes sync metadata to ensure Prefix blocks are **never evicted** from the distributed pool (Tiers 1-4).
*   **Verification:** 8-Node stress tests showed **100% retention** of shared prefixes (10/10) even under memory pressure.

### 2. 🌍 Distributed Content-Addressed Deduplication (Novelty 2)
*   **The Problem:** A popular chatbot service may store the same "You are a helpful assistant..." prompt 10,000 times.
*   **Our Solution:** **Global SHA256-based Deduplication**.
    *   Data is hashed (`SHA256(Block)`) to generate a unique ID.
    *   A **Distributed Hash Table (DHT)** maps `HashID` → `PhysicalLocation`.
    *   Subsequent writes of the same content are **instantly acknowledged** without data transfer.
*   **Result:** **20 Dedup Hits** recorded in validation test, saving redundant transfers across ranks.

### 3. 📍 Locality-Aware Hierarchical Placement (Novelty 3)
*   **The Problem:** Fetching data from a remote node (Tier 3) is faster than disk but slower than local memory.
*   **Our Solution:** **Dynamic Promotion**.
    *   Cascade tracks access frequency for every block.
    *   **Hot Threshold:** If a remote block is accessed >3 times, it is **promoted** to Local GPU/DRAM.
    *   **Cold Demotion:** Rarely used blocks are demoted to Lustre.
*   **Result:** Verified via metadata sync every 100 operations across the cluster.

---

## 📊 Evaluation & Performance Analysis (Updated Feb 14, 2026)

### Experimental Setup
*   **System:** NERSC Perlmutter (HPE Cray EX)
*   **Nodes:** 8 Nodes (32x NVIDIA A100-40GB GPUs)
*   **Interconnect:** Slingshot-11 (200 Gbps)
*   **Workload:** LLaMA-3 70B Int8 KV Blocks (160KB size).

### 📈 1. Weak Scaling (Throughput)
*   **Scenario:** Fixed workload per node (Node count increases, Total data increases).
*   **Objective:** Validate system stability and linear throughput scaling.
*   **Nodes:** 1 → 8 nodes (4 → 32 ranks).

| Nodes | Total GPUs | Throughput (Write) | Scaling Efficiency |
| :---: | :---: | :---: | :---: |
| **1** | 4 | 0.51 GB/s | 100% (Baseline) |
| **2** | 8 | 0.98 GB/s | 96% |
| **4** | 16 | 1.95 GB/s | 96% |
| **8** | 32 | **3.90 GB/s** | **96%** |

> **Analysis:** Cascade demonstrates **near-perfect linear scaling (96% efficiency)** for write-heavy workloads, proving that distributed coordination overhead (DHT, consistency) is negligible even at 8-node scale.

### ⏱️ 2. Strong Scaling (Latency)
*   **Scenario:** Fixed total dataset (**12.21 GB**, ~80k blocks). Node count increases.
*   **Objective:** Verify latency reduction (Speedup) as resources are added.

| Nodes | Write Time (s) | **Speedup (Write)** | Read Time (s) | **Speedup (Read)** | Agg. Read BW |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 6.35 | 1.0x | 0.55 | 1.0x | 22.33 GB/s |
| **2** | 3.03 | 2.1x | 0.27 | 2.0x | 44.77 GB/s |
| **4** | 1.51 | 4.2x | 0.14 | 3.9x | 88.70 GB/s |
| **8** | **0.79** | **8.0x** | **0.08** | **6.8x** | **160.65 GB/s** |

> **Analysis:**
> *   **Perfect Linear Speedup (8.0x):** Write latency decreases exactly in proportion to the node count.
> *   **Massive Read Bandwidth (160 GB/s):** By aggregating GPU HBM and DRAM across 8 nodes, Cascade serves the fixed dataset at memory-bandwidth speeds, eliminating I/O bottlenecks.

### Feature Verification
*   **Dedup Efficiency:** 20 identical system prompt blocks resulted in **20 Dedup Hits** (0 bytes written).
*   **Eviction Policy:** Under 5MB memory constraint, 10/10 Prefix blocks were preserved, while 100% of Suffix blocks were correctly evicted to Lustre.

---

## 🔧 Installation & Usage

### Prerequisites
*   Linux (Cray OS or Ubuntu)
*   NVIDIA CUDA 12.x
*   MPI (Cray MPICH or OpenMPI) w/ CUDA-aware support
*   Python 3.10+

### Step 1: Clone & Build
```bash
git clone https://github.com/changjongkim/kcj-cascade-v6.git
cd kcj-cascade-v6
./build_cpp.sh  # Compiles Backend (src/cascade_backend.cpp)
```

### Step 2: Running a Benchmark
```bash
# Submit a scaling test to SLURM (e.g., 4 nodes)
cd benchmark/scripts
sbatch -N 4 v6_distributed_bench.slurm
```

### Step 3: Python API Example
```python
from cascade_cpp import DistributedStore, CascadeConfig

# 1. Initialize with V6 Features Enabled
cfg = CascadeConfig()
cfg.dedup_enabled = True
cfg.semantic_eviction = True
cfg.locality_aware = True
store = DistributedStore(cfg)

# 2. Put Data (Auto-Tiering + Dedup)
# 'is_prefix=True' marks this as a protected block
store.put("sys_prompt_v1", data_tensor.numpy(), is_prefix=True)

# 3. Get Data (Transparent Retrieval from any Tier)
result = np.empty_like(data_tensor)
store.get("sys_prompt_v1", result)
```

---

## 📂 Repository Structure
```
kcj-cascade-v6/
├── cascade_Code/
│   └── cpp/src/
│       ├── gpu_backend.cu       # Tier 1 (HBM) Manager
│       ├── distributed_backend.cpp # Tier 3/4 (Remote) Manager
│       └── global_dedup.cpp     # Novelty 2: Distributed Hash Table
├── benchmark/
│   ├── scripts/                 # SLURM Job Scripts (Scaling, Etc)
│   ├── data_external/           # ShareGPT, PG-19 Data
│   └── v6_distributed_bench.py  # Main Verification Script
└── docs/                        # Experimental Logs & Reports
```

---
