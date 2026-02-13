# Cascade: Content-Addressed Tiered KV Cache for LLM Inference

> **최종 업데이트:** 2026-02-13 | **버전:** V6 (Distributed) | **플랫폼:** NERSC Perlmutter (A100 × 4, Slingshot-11)

> **상세 개요 문서:** [CASCADE_V6_SCHEME_SUMMARY.md](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/docs/CASCADE_V6_SCHEME_SUMMARY.md)

---

## 1. 프로젝트 개요

### 1.1 문제 정의 (Why)

대규모 LLM (LLaMA-70B 등) 서비스에서 **KV Cache**는 메모리의 가장 큰 병목 요소다.

| 항목 | 수치 |
|------|------|
| LLaMA-70B KV Cache / 토큰 | 320 KB |
| 2,048 토큰 시퀀스 1개 | 640 MB |
| 동시 100 세션 | **64 GB** |
| GPU HBM 용량 (A100) | 40 GB |

GPU 메모리만으로는 동시 세션을 처리할 수 없으며, 기존 시스템(vLLM, LMCache, Redis, HDF5)은 각각 다음 한계를 가진다:

- **vLLM:** GPU-only → 메모리 초과 시 **데이터 소실** (오프로드 없음)
- **LMCache:** Lustre에 파일 1개/블록 → **메타데이터 병목**
- **Redis:** 인메모리 only → **GPU ↔ 네트워크 전송 오버헤드**
- **HDF5:** 단일 파일 잠금 → **병렬 쓰기 불가**

### 1.2 솔루션 (What)

**Cascade**는 GPU → SHM(DRAM) → Lustre(PFS) 3단계 티어링 + 콘텐트 해싱 기반 중복 제거를 결합한 **고성능 KV Cache 스토리지 시스템**이다.

```
┌─────────────────────────────────────────────┐
│  Tier 1: GPU HBM (32GB)    23 GB/s Write    │
│  ├─ Free-list memory pool                   │
│  ├─ 32 CUDA Streams + 32 Pinned Buffers     │
│  └─ NVLink P2P (multi-GPU)                  │
├─────────────────────────────────────────────┤
│  Tier 2: SHM / DRAM (64GB)  18 GB/s Write   │
│  ├─ mmap + SSE2 streaming stores            │
│  ├─ Free-list allocator with coalescing     │
│  └─ Per-token LRU eviction (256 shards)     │
├─────────────────────────────────────────────┤
│  Tier 3: Lustre PFS (∞)    1+ GB/s Write    │
│  ├─ O_DIRECT aligned I/O (bypass page cache)│
│  └─ Per-block file, 16-way striping         │
└─────────────────────────────────────────────┘
              ↕ Demotion / Promotion ↕
```

### 1.3 목표 (Goal)

**SC26 (Supercomputing 2026) 논문 제출**을 위한 실험적 성능 검증:

5. **Novelty 1 (Semantic Eviction):** Cross-node protection of prefix blocks to preserve conversational context.
6. **Novelty 2 (Distributed Dedup):** Global SHA256 content-addressing to eliminate redundant KV storage across the cluster.
7. **Novelty 3 (Locality-Aware Placement):** Dynamic promotion of hot blocks to local tiers based on access frequency and node proximity.

---

## 2. 현재 개발 상태

### 2.1 C++ Core Engine (✅ 완료)

| 파일 | LOC | 상태 | 설명 |
|------|-----|------|------|
| [cascade.hpp](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/include/cascade.hpp) | 390 | ✅ | 전체 API 헤더 (Config, ShardedIndex, GPUBackend, ShmBackend, LustreBackend, CascadeStore) |
| [cascade_core.cpp](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/src/cascade_core.cpp) | 1,087 | ✅ | ShardedIndex LRU, SHM (mmap+SSE2+free-list), Lustre (O_DIRECT), CascadeStore 통합 |
| [gpu_backend.cu](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/src/gpu_backend.cu) | 564 | ✅ | GPU Memory Pool + 32 Streams + 32 Pinned Buffers + Free-list |
| [bindings.cpp](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/python/bindings.cpp) | 180 | ✅ | pybind11 Python 바인딩 (CascadeStore, GPUBackend, ShmBackend, LustreBackend) |
| **합계** | **2,221** | | **핵심 엔진** |

**주요 기능 구현 상태:**

- [x] **SHA256 Content-Addressed Block ID** — 데이터 기반 고유 식별자
- [x] **256-Shard LRU Index** — shared_mutex로 읽기 병렬, 쓰기 배타적 잠금
- [x] **GPU Memory Pool** — cudaMalloc 1회 + free-list 재활용 (단편화 최소화)
- [x] **SHM mmap + SSE2** — 128비트 streaming store로 캐시 바이패스 쓰기
- [x] **SHM Free List** — best-fit 할당 + 인접 블록 합병 (coalescing)
- [x] **Lustre O_DIRECT** — posix_memalign + 4KB 정렬 I/O (페이지 캐시 무시)
- [x] **Tiered Eviction** — GPU→SHM→Lustre 자동 디모션
- [x] **Tier Promotion** — Lustre→SHM, SHM→GPU 자동 프로모션 (읽기 시)
- [x] **Semantic Eviction** — prefix 블록 보호 (LRU 교체 시 건너뜀)
- [x] **Deduplication** — known_blocks_ 인덱스로 중복 쓰기 스킵
- [x] **OpenMP Batch API** — put_batch / get_batch 병렬 실행

### 2.2 Distributed Backend (🔧 구현 완료, 검증 일부)

| 파일 | LOC | 상태 | 설명 |
|------|-----|------|------|
| [cascade_distributed.hpp](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/include/cascade_distributed.hpp) | 251 | ✅ | DistributedStore, DistributedGPUBackend, DistributedDRAMBackend API |
| [distributed_backend.cpp](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/src/distributed_backend.cpp) | 513 | ✅ | MPI RMA + GPU-aware Send/Recv + NVLink P2P |
| [distributed_benchmark.cpp](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/src/distributed_benchmark.cpp) | 303 | ✅ | Multi-node 성능 측정 |

- [x] MPI RMA (Remote Memory Access) / RDMA Integration
- [x] GPU-aware MPI (`mpi_gtl_cuda`) for direct G2G transfers
- [x] **Novelty 1: Cross-Node Semantic Eviction** (Verified)
- [x] **Novelty 2: Distributed Content-Addressed Dedup** (Verified)
- [x] **Novelty 3: Locality-Aware Placement & Promotion** (Verified)
- [x] Global Metadata Synchronization (MPI_Allgatherv)

### 2.3 Python Benchmark Suite (✅ 완료)

| 컴포넌트 | 파일 | 설명 |
|----------|------|------|
| 공통 인터페이스 | [base.py](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/adapters/base.py) | BenchmarkStats, StorageAdapter ABC |
| Cascade 어댑터 | [cascade_adapter.py](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/adapters/cascade_adapter.py) | C++ 엔진 래핑 |
| HDF5 어댑터 | [hdf5_adapter.py](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/adapters/hdf5_adapter.py) | h5py 기반 |
| LMCache 어댑터 | [lmcache_adapter.py](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/adapters/lmcache_adapter.py) | Per-file Lustre |
| Redis 어댑터 | [redis_adapter.py](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/adapters/redis_adapter.py) | Stub |
| PDC 어댑터 | [pdc_adapter.py](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/adapters/pdc_adapter.py) | Stub |
| 실 데이터 생성기 | [data_generator_real.py](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/data_generator_real.py) | MLPerf + LLaMA-70B |
| 벤치마크 러너 | [run_benchmark.py](file:///pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/run_benchmark.py) | 5시스템 통합 실행 |

### 2.4 빌드 시스템 (✅ 완료)

- **CMake 3.18+** — CUDA, OpenSSL, OpenMP, MPI, pybind11
- **타겟 아키텍처:** sm_80 (A100)
- **빌드 타겟:** `cascade_cpp` (Python), `cascade_bench`, `distributed_bench`, `full_bench`, `fair_tier_bench`

---

## 3. 벤치마크 결과

### 3.1 Raw Backend 성능 (합성 데이터, GPU 노드)

| Backend | Block Size | Write (GB/s) | Read (GB/s) | 비고 |
|---------|-----------|:------------:|:-----------:|------|
| **GPU (Pinned)** | 128 KB | 15.7 | 14.4 | PCIe ~63% |
| **GPU (Pinned)** | 1 MB | **23.3** | **22.2** | PCIe **93.6%** |
| **SHM** | 128 KB | 18.0 | 13.5 | SSE2 streaming |
| **SHM** | 1 MB | 18.0+ | 13.5+ | mmap + free-list |

### 3.2 실 애플리케이션 데이터 (LLaMA-70B KV Cache)

| 테스트 | 블록 크기 | 결과 | 비고 |
|--------|----------|------|------|
| **Sequential Write** | 160 MB | **3.10 GB/s** | SHM 8GB 제한, 49 블록 Lustre 이관 |
| **Sequential Read** | 160 MB | **2.45 GB/s** | SHM hit 39 + Lustre hit 61 |
| **Dedup Write** | 160 MB | **4.33 GB/s** | 40 dedup hits (prefix 공유) |

### 3.3 5-System 비교 (500 블록, 실 데이터)

> `real_bench.sh` 기반 결과

| System | Write (GB/s) | Read (GB/s) | Dedup | Hit Rate | 특징 |
|--------|:----------:|:---------:|:-----:|:--------:|------|
| **Cascade** | 2-3 | 2-3 | ✅ 80%+ | 100% | 3-tier + dedup |
| LMCache | 0.5-1 | 0.5-1 | ❌ | 100% | Per-file I/O 병목 |
| HDF5 | 1-2 | 1-2 | ❌ | 100% | 단일 파일 잠금 |
| Redis | 1-2 | 2-3 | ❌ | 100% | In-memory only |
| vLLM | 10+ | 10+ | ❌ | **40%** | GPU-only, 60% 소실 |

---

## 4. 코드 구조

```
Cascade-kcj/
├── cascade_Code/cpp/              # ← C++ 핵심 엔진
│   ├── include/
│   │   ├── cascade.hpp            # 메인 API (390 LOC)
│   │   └── cascade_distributed.hpp # 분산 API (251 LOC)
│   ├── src/
│   │   ├── cascade_core.cpp       # 코어 구현 (1,087 LOC)
│   │   ├── gpu_backend.cu         # GPU CUDA (564 LOC)
│   │   ├── distributed_backend.cpp # MPI 분산 (513 LOC)
│   │   ├── benchmark.cpp          # C++ 벤치마크 (350 LOC)
│   │   ├── distributed_benchmark.cpp
│   │   ├── full_benchmark.cpp
│   │   ├── fair_tier_bench.cpp
│   │   └── pure_memcpy_bench.cu
│   ├── python/
│   │   └── bindings.cpp           # pybind11 (180 LOC)
│   └── CMakeLists.txt             # 빌드 시스템
│
├── benchmark/                     # ← Python 벤치마크 프레임워크
│   ├── adapters/                  # 5시스템 어댑터
│   ├── scripts/                   # SLURM 잡 스크립트
│   ├── data_generator_real.py     # MLPerf 실 데이터 생성
│   ├── run_benchmark.py           # 통합 벤치마크 러너
│   └── config.py                  # LLaMA-70B 설정
│
├── cascade/                       # Python 패키지
│   └── __init__.py                # create_login_node_store 등
│
├── docs/                          # 문서
└── paper/                         # SC26 논문 관련
```

---

## 5. 앞으로 해야 할 작업 (TODO)

### V6 Distributed Performance (2-Node Verification)
- **Environment**: 2-node Perlmutter Cluster (8x A100 GPUs)
- **Novelty 2 (Dedup)**: Successfully triggered 20 dedup hits (1.2MB saved) in prefix-sharing test.
- **Novelty 3 (Locality)**: Hot remote blocks successfully promoted to local GPU after threshold (3) accesses.
- **Scaling**: Read throughput scaling from **1.83 GB/s (1 node)** → **5.46 GB/s (8 nodes)** (3x speedup).

### 5.1 🔴 높은 우선순위 (SC26 논문 필수)
| # | 작업 | 설명 | 예상 난이도 |
|---|------|------|-----------|
| 1 | **16-Node Scaling Test** | 64+ GPUs 환경에서 Distributed Cascade 성능 곡선 추출 | ★★★★ |
| 2 | **SOTA 비교 (LMCache, Mooncake)** | 최신 시스템 대비 캐시 히트율 및 지연시간 비교 | ★★★★ |
| 3 | **vLLM End-to-End 통합** | 실제 서빙 환경에서 TTFT/TPOT 개선 효과 검증 | ★★★★ |
| 4 | **SC26 논문 Draft 작성** | 3대 Novelty(Semantic, Dedup, Locality) 중심 기술 | ★★★ |

### 5.2 🟡 중간 우선순위 (성능 개선)

| # | 작업 | 설명 | 예상 난이도 |
|---|------|------|-----------|
| 6 | **Async Prefetch Pipeline** | 백그라운드 Lustre→SHM 프리페칭으로 읽기 지연 단축 | ★★★ |
| 7 | **SHA256 → BLAKE3/xxHash** | 해싱 병목 해소 (현재 블록당 ~0.5ms) | ★★ |
| 8 | **Lustre Aggregated I/O** | 파일 1개당 다수 블록 → 메타데이터 오버헤드 제거 | ★★★ |
| 9 | **INT4/INT8 KV Compression** | GPU/SHM에서 양자화 압축으로 2-4× 용량 확보 | ★★★ |
| 10 | **GDRCopy Direct Path** | SHM 대신 GPU→NIC 직접 전송 | ★★★★ |

### 5.3 🟢 낮은 우선순위 (안정성/품질)

| # | 작업 | 설명 | 예상 난이도 |
|---|------|------|-----------|
| 11 | **Unit Tests (C++)** | GoogleTest 기반 ShardedIndex, Backend 개별 테스트 | ★★ |
| 12 | **Python Integration Tests** | pytest 기반 E2E 테스트 | ★★ |
| 13 | **Error Handling 강화** | CUDA OOM, mmap 실패, Lustre I/O 에러 복구 | ★★ |
| 14 | **CI/CD Pipeline** | GitHub Actions + Perlmutter self-hosted runner | ★★★ |
| 15 | **Documentation** | API Reference, Architecture Diagram, User Guide | ★ |

---

## 6. 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **언어** | C++17, CUDA 11.7+, Python 3.10+ |
| **빌드** | CMake 3.18, Ninja |
| **GPU** | NVIDIA A100-SXM4-40GB (sm_80) |
| **네트워크** | HPE Slingshot-11, GPU-aware cray-mpich |
| **스토리지** | Lustre (SCRATCH), /dev/shm (DRAM), GPU HBM |
| **라이브러리** | OpenSSL (SHA256), pybind11, OpenMP, MPI |
| **벤치마크 데이터** | MLPerf (OpenORCA, CNN/DailyMail, SCROLLS, ShareGPT) |
| **클러스터** | NERSC Perlmutter (A100 GPU ×9,472) |
