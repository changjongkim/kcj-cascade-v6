# 🚀 Cascade: HPC-Scale KV Cache Storage for LLM Inference

[![SC'26](https://img.shields.io/badge/Target-SC'26-blue.svg)](https://supercomputing.org/)
[![Perlmutter](https://img.shields.io/badge/Platform-NERSC%20Perlmutter-green.svg)](https://docs.nersc.gov/systems/perlmutter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cascade** is a **4-tier hierarchical KV cache storage system** designed for HPC-scale LLM inference. It achieves **1.77× higher read throughput** than LMCache and **eliminates the 85% data loss** observed in GPU-only systems like vLLM.

> 📝 **Paper Status**: SC'26 submission in progress

---

## 🎯 The Problem

LLM inference is **memory-bound**: 80% of time is spent loading KV cache from memory. Current solutions fail at HPC scale:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WHY EXISTING SYSTEMS FAIL                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  vLLM (GPU-only):     ████████████████████████████░░░░  85% DATA LOSS!     │
│                       Only 30 blocks fit in GPU, 170 blocks EVICTED         │
│                       → Forces expensive KV recomputation                   │
│                                                                             │
│  LMCache (per-file):  ████████░░░░░░░░░░░░░░░░░░░░░░░░  Metadata overhead  │
│                       Creates thousands of small files on Lustre            │
│                       → 0.2-1.3 GB/s (10× slower than aggregated I/O)       │
│                                                                             │
│  Redis (in-memory):   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Network bottleneck │
│                       Serialization + TCP overhead                          │
│                       → 1.22 GB/s read (6× slower than Lustre)              │
│                                                                             │
│  HDF5 (single-file):  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░  No tiered caching  │
│                       All I/O goes to Lustre                                │
│                       → 3.38 GB/s read (2× slower than Cascade)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 Core Contributions

### 1. **Tiered Storage Hierarchy** (GPU → SHM → Lustre)

```
                    BANDWIDTH COMPARISON
     ┌────────────────────────────────────────────────────────┐
     │                                                        │
  GPU│████████████████████████████████████████  1,555 GB/s   │ ← Hot data
 HBM │                                                        │
     │                                                        │
 SHM │████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░    33-45 GB/s │ ← Warm data
DRAM │                                                        │
     │                                                        │
Lus- │██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    6.8-8 GB/s │ ← Cold data
tre  │                                                        │
     │                                                        │
Per- │█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.2-1.3 GB/s │ ← LMCache
file │                                                        │
     └────────────────────────────────────────────────────────┘
       0        500       1000      1500     GB/s
```

**Why it matters:** vLLM loses 85% of data because it only uses GPU. Cascade spills to SHM (46× faster than Lustre) and Lustre (persistent), retaining **100% of data**.

### 2. **Content-Addressed Deduplication**

```python
# Session-based ID (LMCache, vLLM):       Content-based ID (Cascade):
block_id = f"session_{user_id}_{seq}"  →  block_id = sha256(key + value)[:32]

# Result:
# 50 users × same system prompt = 50 blocks    →    1 block (98% reduction)
```

**Why it matters:** In LLM serving, system prompts are shared across users. With 50 sessions using the same prompt, Cascade stores **1 block** instead of 50.

### 3. **Aggregated Lustre I/O**

```
LMCache:   session_001_block_000.bin   ──┐
           session_001_block_001.bin     │──→ 3,200 files = 3,200 metadata ops
           session_002_block_000.bin     │
           ...                         ──┘

Cascade:   agg_rank000_000000.bin  ───────→ 16 files = 16 metadata ops (200× less)
           agg_rank001_000000.bin
           ...
```

**Why it matters:** Lustre metadata operations are expensive. Cascade uses `lfs setstripe -c 16 -S 4m` for optimal striping.

---

## 📊 Benchmark Results (4 nodes, 16 GPUs, 530GB data)

```
                    READ THROUGHPUT COMPARISON
     ┌────────────────────────────────────────────────────────┐
     │                                                        │
Casc-│████████████████████████████████████░░░░░░  7.16 GB/s  │ ⭐ BEST
ade  │  GPU(30) + SHM(50) + Lustre(120) = 100% retention     │
     │                                                        │
LM-  │████████████████████░░░░░░░░░░░░░░░░░░░░░░  4.04 GB/s  │
Cache│  Per-file Lustre I/O                                   │
     │                                                        │
HDF5 │██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  3.38 GB/s  │
     │  Single file, no tiering                               │
     │                                                        │
Redis│██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.22 GB/s  │
     │  Network serialization bottleneck                      │
     │                                                        │
vLLM │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  N/A        │ ❌ 85% LOST
     │  170/200 blocks evicted!                               │
     └────────────────────────────────────────────────────────┘
       0    1    2    3    4    5    6    7    8  GB/s
```

### Key Metrics

| System | Read (GB/s) | Write (GB/s) | Hit Rate | Data Loss |
|--------|-------------|--------------|----------|-----------|
| **Cascade** | **7.16** | 0.44 | **100%** | 0 blocks |
| LMCache | 4.04 | 0.50 | 100% | 0 blocks |
| HDF5 | 3.38 | 0.50 | 100% | 0 blocks |
| Redis | 1.22 | 0.67 | 100% | 0 blocks |
| vLLM | — | 0.99 | **15%** | **170 blocks (85%)** |

### Why Cascade Wins

1. **1.77× faster read** than LMCache: GPU+SHM tiers serve 40% of requests (80/200 blocks)
2. **100% vs 15% hit rate** vs vLLM: Tiered overflow prevents eviction
3. **Zero data loss**: All 200 blocks preserved across GPU(30) + SHM(50) + Lustre(120)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                               │
│                         (vLLM, LMCache, custom inference)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CASCADE STORE                                   │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────────────────┐ │
│  │ Dedup Index  │  │  Tier Manager  │  │    Semantic Eviction Policy     │ │
│  │  (SHA-256)   │  │  (GPU→SHM→L)   │  │  (LRU + prefix-aware + refcnt)  │ │
│  └──────────────┘  └────────────────┘  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                        │
          ▼                    ▼                        ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐
│   GPU Backend    │  │   SHM Backend    │  │      Lustre Backend          │
│   (CUDA malloc)  │  │   (mmap /dev/shm)│  │  (Aggregated + Striped I/O)  │
│   1,555 GB/s     │  │   33-45 GB/s     │  │  6.8-8 GB/s (aggregated)     │
│   40GB × 4 GPUs  │  │   128 GB/node    │  │  44 PB (Perlmutter $SCRATCH) │
└──────────────────┘  └──────────────────┘  └──────────────────────────────┘
```

### Memory Hierarchy Details

| Tier | Capacity | Bandwidth | Latency | Use Case |
|------|----------|-----------|---------|----------|
| **GPU HBM** | 40GB × 4 = 160GB/node | 1,555 GB/s | μs | Hot KV cache (active inference) |
| **Shared Memory** | 128GB/node (/dev/shm) | 33-45 GB/s | μs | Warm data (recently evicted) |
| **Lustre PFS** | 44PB ($SCRATCH) | 6.8-8 GB/s | 10s ms | Cold data (persistent storage) |
| **Per-file Lustre** | — | 0.2-1.3 GB/s | 100s ms | ❌ Avoid (LMCache pattern) |

---

## 🚀 Quick Start (NERSC Perlmutter)

### Prerequisites

### Prerequisites

```bash
# Login to Perlmutter
ssh <username>@perlmutter.nersc.gov

# Clone repository
cd $SCRATCH
git clone https://github.com/sunggonkim/Cascade.git
cd Cascade
```

### Environment Setup

```bash
# Load required modules
module load python cudatoolkit cray-mpich libfabric

# Set environment variables
export SCRATCH=/pscratch/sd/s/sgkim
export CASCADE_HOME=$SCRATCH/Cascade
export PYTHONPATH=$CASCADE_HOME:$PYTHONPATH
```

### Build C++ Components

```bash
cd cascade_Code/cpp
./build_perlmutter.sh
```

### Run Benchmark (Debug Queue - 4 nodes max)

```bash
# Submit 4-node benchmark (max for debug queue)
sbatch benchmark/scripts/max_debug_bench.sh

# Check results
cat benchmark/logs/max_debug_<jobid>.out
```

---

## 📁 Repository Structure

```
Cascade/
├── cascade_Code/              # Core implementation
│   └── cpp/                   # C++ with CUDA
│       ├── src/cascade_core.cpp    # Tiered store logic
│       └── src/gpu_backend.cu      # GPU memory management
│
├── benchmark/                 # Benchmark framework
│   ├── adapters/             # Storage system adapters
│   │   ├── cascade_adapter.py
│   │   ├── lmcache_adapter.py
│   │   └── ...
│   └── scripts/              # SLURM scripts
│       └── max_debug_bench.sh
│
├── paper/                    # SC'26 LaTeX paper
│   ├── main.tex
│   ├── 4. Evaluation.tex    # ← Results
│   └── Figures/
│
├── third_party/              # Baseline implementations
│   ├── LMCache/             # State-of-the-art KV cache
│   ├── vllm/                # PagedAttention reference
│   └── redis/               # In-memory store
│
└── docs/                     # Documentation
    ├── BENCHMARK.md
    ├── DEVELOPMENT.md
    └── PAPER.md
```

---

## 📈 Running Production Benchmarks

### 1. Generate KV Cache Data (500GB)

```bash
# Generate realistic LLaMA-70B KV cache data
sbatch benchmark/scripts/generate_data.sh
# Output: $SCRATCH/cascade_kv_cache/ (3,200 blocks × 168MB)
```

### 2. Run Multi-System Comparison

```bash
# Full benchmark: Cascade, vLLM, LMCache, HDF5, Redis
sbatch benchmark/scripts/max_debug_bench.sh
# Results: benchmark/results/max_debug_<jobid>_rank*.json
```

### 3. View Results

```bash
# Check job output
cat benchmark/logs/full_6sys_<jobid>.out

# Parse JSON results
python -c "
import json
with open('benchmark/results/full_6sys_<jobid>.json') as f:
    data = json.load(f)
    for sys, res in data.items():
        print(f'{sys}: {res[\"write_gbps\"]:.2f} GB/s write, {res[\"read_gbps\"]:.2f} GB/s read')
"
```

---

## 🛠️ Development Guide

### Adding a New Storage Backend

1. Create adapter in `benchmark/adapters/`:

```python
# benchmark/adapters/my_adapter.py
from .base import StorageAdapter

class MyAdapter(StorageAdapter):
    def __init__(self, config):
        super().__init__("MySystem", config)
    
    def initialize(self) -> bool:
        # Setup your storage system
        return True
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        # Store block
        return True
    
    def get(self, block_id: str) -> Optional[tuple]:
        # Retrieve block
        return (key_data, value_data)
```

2. Register in `benchmark/adapters/__init__.py`

3. Add to benchmark script

### Block ID Convention

**CRITICAL**: All block IDs must be content-addressed:

```python
import hashlib

def compute_block_id(key: bytes, value: bytes) -> str:
    h = hashlib.sha256()
    h.update(key)
    h.update(value)
    return h.hexdigest()[:32]
```

---

## 📝 Paper Workflow

### Building the Paper

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Updating Results

After running benchmarks:

1. Parse results from `benchmark/results/`
2. Update numbers in `4. Evaluation.tex`
3. Regenerate figures: `python generate_figures.py`
4. Rebuild PDF

### Key Files to Update

| Section | File | What to Update |
|---------|------|----------------|
| Intro | `1. Introduction.tex` | Headline numbers |
| Eval | `4. Evaluation.tex` | Tables, figures, analysis |
| Figures | `Figures/` | TikZ charts, diagrams |

---

## 📊 LLaMA-70B KV Cache Dimensions

| Parameter | Value |
|-----------|-------|
| Layers | 80 |
| KV Heads (GQA) | 8 |
| Head Dimension | 128 |
| Dtype | float16 |
| **Per Token** | 2 × 80 × 8 × 128 × 2 = **320 KB** |
| **Per Block (256 tokens)** | 256 × 320 KB = **~168 MB** |

---

## 🧪 Baseline Systems

| System | Source | Purpose |
|--------|--------|---------|
| **vLLM** | `third_party/vllm/` | GPU-only PagedAttention |
| **LMCache** | `third_party/LMCache/` | State-of-the-art KV cache |
| **HDF5** | h5py (pip) | Standard HPC I/O |
| **Redis** | `third_party/redis/` | In-memory key-value |
| **PDC** | `third_party/pdc/` | HPC object storage |

### Building Baselines

```bash
# Redis
cd third_party/redis && make

# Mercury (required for PDC)
cd third_party/mercury
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install \
         -DNA_USE_OFI=ON -DNA_OFI_TESTING_PROTOCOL=tcp
make -j8 && make install

# PDC
cd third_party/pdc
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install \
         -DMERCURY_DIR=../../mercury/install
make -j8 && make install
```

---

## 🔧 Troubleshooting

### Common Issues

**Import Error: cascade module not found**
```bash
export PYTHONPATH=/path/to/Cascade:$PYTHONPATH
```

**Redis connection refused**
```bash
# Check Redis is running
$CASCADE_HOME/third_party/redis/src/redis-cli -p 6380 ping
```

**PDC server not starting**
```bash
# Check Mercury installation
ldd $CASCADE_HOME/third_party/pdc/install/bin/pdc_server
```

**Lustre quota exceeded**
```bash
# Check usage
lfs quota -u $USER $SCRATCH
# Clean old data
rm -rf $SCRATCH/cascade_kv_cache_old/
```

---

## 📚 Citation

```bibtex
@inproceedings{cascade2026,
  title={Cascade: HPC-Scale KV Cache Storage for LLM Inference},
  author={Kim, Sung Gon},
  booktitle={SC'26: International Conference for High Performance Computing, 
             Networking, Storage and Analysis},
  year={2026}
}
```

---

## 📧 Contact

- **Author**: Sung Gon Kim
- **Email**: sgkim@lbl.gov
- **Institution**: Lawrence Berkeley National Laboratory / NERSC

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
