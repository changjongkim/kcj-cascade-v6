# Cascade V6: Content-Addressed Distributed Tiered KV Cache (SC26)

## 1. Goal (Vision)
**To solve the Memory Wall problem in LLM Serving** by creating a **Distributed, 5-Tier, Content-Addressed KV Cache** that utilizes **all available memory** across the cluster (GPU HBM + Host DRAM + Distributed DRAM + Lustre PFS).

**Primary Target:** SC26 (Supercomputing 2026) Paper Submission
**Baseline Comparisons:** vLLM (GPU-only), LMCache (Lustre-based), Redis (In-memory KV), HDF5 (File-based).

---

## 2. Core Problem & Solution
*   **Problem:** LLM inference requires massive KV cache memory (e.g., LLaMA-70B: ~320KB/token). GPU memory is limited (40-80GB), causing frequent re-computation or request failures.
*   **Cascade Solution:** A **Disaggregated Memory Store** that seamlessly spills over data to cheaper tiers (DRAM, Remote Memory, SSD/Lustre) while maintaining high throughput via **Content-Addressing** and **Novel Caching Policies**.

---

## 3. The 3 Core Novelties (SC26 Contributions)

### **Novelty 1: Cross-Node Semantic-Aware Eviction**
*   **Concept:** Not all KV blocks are equal. System prompts (prefixes) are critical for context but rarely change.
*   **Mechanism:**
    *   Protect **Prefix Blocks** globally across the cluster.
    *   Evict **Least Recently Used (LRU)** non-prefix blocks first.
    *   **Result:** Maintains high cache hit rate for shared contexts even under memory pressure.
    *   **Verification:** Confirmed 100% prefix retention in 2-node stress tests.
    *   **Status:** ✅ Fully Implemented & Verified.

### **Novelty 2: Distributed Content-Addressed Deduplication**
*   **Concept:** Identical text generates identical KV blocks (based on SHA256 hash).
*   **Mechanism:**
    *   **Global Index:** All nodes share knowledge of block locations.
    *   **Dedup Write:** If *Hash(A)* exists on Node 1, Node 2 does *not* write it again. It just references Node 1.
    *   **Result:** Eliminates redundant storage, increasing effective cluster capacity.
    *   **Verification:** Confirmed 20/20 dedup hits (1.2MB saved) in multi-rank experiments.
    *   **Status:** ✅ Fully Implemented & Verified.

### **Novelty 3: Locality-Aware Hierarchical Placement**
*   **Concept:** Remote access is slower than local access. Hot data should be local.
*   **Mechanism:**
    *   **Access Tracking:** Count frequency of remote reads for each block.
    *   **Auto-Promotion:** If remote access > Threshold (3), migrate block to **Local GPU**.
    *   **Result:** Dynamic optimization of data placement based on access patterns.
    *   **Verification:** Verified automatic promotion of hot blocks from Remote -> Local GPU.
    *   **Status:** ✅ Fully Implemented & Verified.

---

## 4. Architecture: 5-Tier Storage Hierarchy

| Tier | Device | Throughput | Capacity | Logic |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | **Local GPU (HBM)** | ~2 TBytes/s | Small (40-80GB) | Hot / Working Set |
| **L2** | **Local DRAM (System)** | ~25 GBytes/s | Medium (512GB) | Warm / Overflow |
| **L3** | **Remote GPU** | ~25 GBytes/s (NVLink/PCIe) | Cluster-wide | Remote Hot |
| **L4** | **Remote DRAM** | ~10-20 GBytes/s (RDMA) | Large (TBs) | Remote Warm |
| **L5** | **Lustre PFS** | ~1-5 GBytes/s | Infinite (PBs) | Cold / Archive |

*   **Implementation:** C++ Backend with CUDA, MPI (RMA/GTL), and OpenMP.
*   **KV Compression:** INT8 quantization support to double effective capacity.

---

## 5. Experimental Results (Perlmutter A100 Cluster)

### **A. Distributed Scaling (1 to 8 Nodes)**

| Nodes | Total GPUs | Agg. Write (GB/s) | Agg. Read (GB/s) | Speedup (Read) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 4 | 3.10 | 1.83 | 1.0x (Baseline) |
| **2** | 8 | 2.37 | 1.40 | 0.76x |
| **4** | 16 | 3.87 | 2.79 | 1.52x |
| **8** | 32 | **4.24** | **5.46** | **2.98x** |

*   **Observation:** Read throughput scales linearly after initial multi-node overhead (2 nodes).
*   **Implication:** Efficient parallel data serving across the cluster.

### **B. Novelty Verification (2 Nodes)**
1.  **Semantic Eviction:** 10/10 Prefix blocks protected while 40 regular blocks were evicted.
2.  **Dedup:** 20 redundant writes skipped (Saved 1.2 MB space).
3.  **Locality:** 2 hot blocks promoted to local GPU after 3 remote reads.

### **C. Single-Node Raw Performance**
*   **Write:** 23.3 GB/s (93.6% of PCIe Gen4 limit)
*   **Read:** 22.2 GB/s

---

## 6. Next Steps (Roadmap to SC26)

1.  **Large-Scale Scaling:** Test on **16, 32, 64 nodes** to verify weak scaling efficiency.
2.  **vLLM Integration:** Connect Cascade as the backing store for vLLM to measure **TTFT (Time To First Token)** and **TPOT (Time Per Output Token)** improvements in real serving scenarios.
3.  **Comparison:** Direct benchmark vs. **Mooncake** (KVCache-on-paged-memory) and **LMCache** (File-based).
