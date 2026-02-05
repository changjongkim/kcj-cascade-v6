# Cascade 코드 분석 보고서

## 1. 개요
Cascade는 HPC 규모의 LLM 추론을 위해 설계된 4계층 KV 캐시 스토리지 시스템입니다. GPU HBM, 로컬 공유 메모리(SHM), 원격 SHM, Lustre PFS로 구성된 계층 구조를 활용하여 성능을 최적화합니다.

## 2. 아키텍처 분석

### 2.1 핵심 컴포넌트
- **CascadeStore**: 스토리지 시스템과 상호 작용하기 위한 메인 인터페이스입니다.
- **백엔드 계층 구조**:
    - **Tier 1 (GPU HBM)**: `gpu_backend.cu`에서 처리. 직접 포인터 접근 방식 사용.
    - **Tier 2 (Local SHM)**: `cascade_core.cpp`에서 처리. mmap 및 병렬 memcpy 사용.
    - **Tier 3 (Remote SHM)**: `distributed_backend.cpp`에서 처리. MPI RMA 사용.
    - **Tier 4 (Lustre)**: 콜드 스토리지 폴백(Cold storage fallback).

### 2.2 주요 데이터 구조
- **`ShardedIndex<V>`**: 256개의 샤드(shard)를 가진 동시 해시 맵으로, 각 샤드는 `shared_mutex`로 보호됩니다. `known_blocks`, `prefix_blocks` 및 백엔드 인덱스에 사용됩니다.
- **`CascadeConfig`**: 용량, 경로 및 기능 플래그를 위한 구성 구조체입니다.

## 3. 구현 상세

### 3.1 로컬 SHM 관리 (`cascade_core.cpp`)
- **메커니즘**: `mmap`과의 광범위한 상호작용.
- **최적화**:
    - `MAP_POPULATE`를 사용하여 페이지를 프리폴트(pre-fault)합니다.
    - `MAP_POPULATE` 실패 시 일반 `mmap`으로 폴백합니다.
    - `MADV_WILLNEED` 및 `MADV_HUGEPAGE`와 함께 `madvise`를 사용합니다.
    - **스트리밍 저장(Streaming Stores)**: 4KB 이상의 쓰기에 대해 `_mm_stream_si128` (SSE2)를 사용하여 CPU 캐시를 우회합니다.
    - **프리페칭(Prefetching)**: 4KB 이상의 읽기에 대해 `_mm_prefetch` 및 SSE2 로드(`_mm_load_si128`)와 저장을 사용합니다.
- **중복 제거(Dedup)**: 쓰기 전에 `index_.contains(id)`를 확인합니다.

### 3.2 GPU 상호작용 (`gpu_backend.cu`)
- **메모리 풀**: 범프 할당자(bump allocator)가 있는 `GPUMemoryPool`을 사용합니다. 할당은 빠르지만 개별 블록에 대한 **할당 해제 지원이 없습니다** (`clear` 시 `reset`만 가능).
- **동시성(Concurrency)**:
    - **32 CUDA Streams**: 경합을 피하기 위해 스레드당 하나씩 사용.
    - **32 Pinned Buffers**: 페이징 가능한 메모리(pageable memory)의 스테이징을 위해 각 8MB (총 256MB) 사용.
- **최적화**:
    - `cudaPointerGetAttributes`를 확인하여 호스트 메모리가 이미 고정(pinned)되어 있는지 확인합니다. 그렇다면 직접 DMA(`cudaMemcpyAsync`)를 수행합니다.
    - 페이징 가능한 경우 먼저 고정 버퍼(pinned buffer)로 복사한 다음 DMA를 수행합니다.
- **동기화**: `sync_all`은 32개 스트림 모두를 기다립니다.

### 3.3 분산 계층 (`distributed_backend.cpp`)
- **메커니즘**: `MPI_Put` 및 `MPI_Get`을 사용하는 MPI 단방향 통신(RMA).
- **데이터 배치**: 블록 ID 기반의 단순 일관성 해싱(Consistent hashing) (노드 식별용 첫 2바이트, GPU 식별용 다음 2바이트).
- **GPU 인식(GPU-Aware)**: 효율적인 전송을 위해 MPI 윈도우에 고정 메모리(`cudaHostAlloc`)를 사용합니다.
- **전역 인덱스**: `global_index_`는 클러스터 전체의 블록 위치를 추적합니다.

### 3.4 Python 바인딩 (`bindings.cpp`)
- **pybind11**을 사용하여 C++ 클래스를 Python에 노출합니다.
- **상호 운용성**: NumPy 배열을 C++ 포인터(`py::buffer_info`)로 효율적으로 변환하며, 버퍼 유형에 따라 제로 카피(zero-copy)를 허용합니다.
- **배치 API**: 처리량(throughput)에 영향을 줄 수 있는 `put_batch`/`get_batch`를 노출합니다.

## 4. 관찰 및 잠재적 문제

### 4.1 치명적 문제 (Critical Issues)
- **LustreBackend 불일치**: 헤더 `cascade.hpp`와 `cascade_core.cpp`의 주석은 `io_uring`을 언급하지만, 실제 구현은 표준 동기 POSIX I/O(`open`, `read`, `write`)를 사용합니다. `ring_` 멤버는 사용되지 않습니다. **이는 콜드 스토리지의 비동기 I/O 기능이 구현되지 않았음을 시사합니다.**
- **분산 Put 미완성**: `DistributedGPUBackend::put`에서 데이터는 `MPI_Put`을 통해 원격 노드의 스테이징 버퍼로 전송되지만, 수신 측에서 스테이징된 데이터를 GPU로 이동시키는 메커니즘이 없다는 TODO 주석이 있습니다 (`// TODO: Need a protocol for remote node to store from staging to GPU`). **이는 분산 GPU 쓰기가 기능적으로 고장 났거나 불완전할 수 있음을 의미합니다.**

### 4.2 성능 및 설계 우려 사항
- **메모리 관리**:
    - `GPUMemoryPool`은 `clear()`에서만 재설정되는 범프 할당자를 사용합니다. 개별 블록 해제를 지원하지 않습니다. 이는 특정 벤치마크 패턴(채우기, 테스트, 지우기)에는 작동하지만 블록이 지속적으로 축출되고 교체되는 **실제 동적 LRU 캐시에는 적합하지 않습니다**.
    - 수동 `mmap` 관리는 복잡하고 오류가 발생하기 쉽습니다.
- **에러 처리**:
    - 기본적인 에러 처리(`false` 반환 또는 `perror` 출력)만 되어 있습니다. C++ 예외나 Python으로의 견고한 에러 전파가 없습니다.
    - CUDA 에러는 stderr에 출력되지만 모든 경우에 실행 흐름을 안전하게 중단하지 않을 수 있습니다.

### 4.3 결론
Cascade는 인상적인 이론적 대역폭 활용(pinned memory, zero-copy, SSE2)을 갖춘 고성능 설계를 보여줍니다. 그러나 **프로토타입/벤치마크 상태**인 것으로 보입니다. 프로덕션 시스템을 위한 주요 기능(견고한 메모리 회수, 완전한 분산 프로토콜, 진정한 비동기 디스크 I/O)이 누락되었거나 불완전합니다.
