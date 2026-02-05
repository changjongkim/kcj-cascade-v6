# Cascade 정상화 및 개선 계획 (Remediation Plan)

현재 코드는 "벤치마크용 모형" 수준입니다. 이를 실제 작동하는 고성능 스토리지 시스템으로 탈바꿈시키기 위한 3단계 계획을 제안합니다.

## Phase 1: 기능 정상화 (Make it Work)
**목표**: 거짓말을 없애고, 실제로 데이터가 저장되고 조회되게 만든다.

### 1.1 Tier 4 (Lustre) 진짜 스토리지 만들기
- **현재**: `open()` 후 `write()` -> RAM(OS Cache)에 씀.
- **수정**:
    - `O_DIRECT` 플래그 추가 (페이지 캐시 우회).
    - 동기 방식임을 인정하고 `io_uring` 코드는 삭제하거나, 진짜 `io_uring`을 연동.

### 1.2 Tier 3 (Distributed) 데이터 흐름 완성
- **현재**: `MPI_Put`으로 보내고 끝남 (수신 측 처리 로직 없음).
- **수정**:
    - **Receiver Thread 추가**: 원격 노드에서 들어온 데이터를 감시하거나, MPI 신호를 받아 GPU로 옮기는 Polling 로직 구현.
    - **Global Index 동기화**: 데이터 수신 후 "나한테 이 데이터 있다"라고 클러스터에 알리는 메타데이터 전송 로직 추가.

### 1.3 메모리 관리 (Memory Management)
- **현재**: `alloc()`만 있고 `free()`가 없는 일회용.
- **수정**:
    - `GPUMemoryPool`을 **Slab Allocator** 또는 **Bitmap Allocator**로 교체하여 `free()` 구현.
    - 캐시 축출(Eviction) 정책(LRU) 구현을 위한 기반 마련.

---

## Phase 2: 성능 최적화 (Make it Fast)
**목표**: CPU 병목을 제거하여 진짜 H/W 성능을 끌어낸다.

### 2.1 SHA256 제거
- **현재**: 모든 요청마다 무거운 해시 계산.
- **수정**:
    - **XXHash** 같은 초고속 해시로 변경하거나,
    - 상위 레벨(vLLM)에서 이미 유니크한 ID(Request ID + Token Index)를 생성해서 내려주도록 인터페이스 변경.

### 2.2 진짜 Lock-Free 구현
- **현재**: `std::shared_mutex` 사용.
- **수정**:
    - `userspace-rcu` 라이브러리를 쓰거나,
    - **Cuckoo Hash Map** 또는 **Hopscotch Hashing** 같은 진짜 Lock-Free/Wait-Free 자료구조로 교체.

---

## Phase 3: 신뢰성 확보 (Make it Reliable)

### 3.1 정합성 테스트 (Correctness Test)
- **현재**: 속도만 재는 벤치마크.
- **추가**:
    - 데이터 무결성 검증 (Put한 데이터가 Get했을 때 깨지지 않았는지).
    - 멀티 스레드/멀티 노드 환경에서의 Race Condition 테스트.

### 3.2 에러 핸들링
- 기능 실패 시 조용히 `false` 리턴하는 대신, 명확한 에러 코드나 예외 처리.

---

## 사용자 결정 필요 사항
가장 먼저 어디에 집중하시겠습니까?

1.  **기초 공사**: Tier 4(Fake I/O)와 Memory Leak부터 고쳐서 "오래 돌아가는" 시스템 만들기.
2.  **분산 기능**: Tier 3(데이터 증발)를 고쳐서 "확장 가능한" 시스템 만들기.
3.  **구조 변경**: SHA256 제거 및 Lock-Free 교체로 "병목 없는" 구조부터 잡기.
