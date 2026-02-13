# Cascade V6 코드 분석 및 동작 검증 보고서

**작성일:** 2026-02-13
**상태:** 검증 완료 (8노드 확장성 확인)

## 1. 개요
본 보고서는 NERSC Perlmutter 환경에서 구현된 **Cascade V6 (Distributed 5-Tier KV Cache)** 시스템의 코드 분석 결과와 실제 동작 메커니즘을 한국어로 정리한 문서입니다. 최신 빌드 및 실험을 통해 분산 환경에서의 안정성과 선형 확장성이 검증되었습니다.

## 2. 5계층 계층형 메모리 구조 (5-Tier Hierarchy)
Cascade V6는 데이터의 액세스 빈도와 중요도에 따라 5개의 계층에 데이터를 최적으로 배치합니다.

| 계층 | 명칭 | 구현 기술 | 특징 |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Local GPU** | CUDA, GPUMemPool | 가장 빠른 액세스 (NVLink 활용) |
| **Tier 2** | **Local DRAM** | Pinned Memory | GPU 부족 시 1차 배후지 |
| **Tier 3** | **Remote GPU** | GPU-aware MPI RMA | 타 노드 GPU 메모리 직접 접근 |
| **Tier 4** | **Remote DRAM** | MPI One-sided (RMA) | 클러스터 전체 메모리 풀 활용 |
| **Tier 5** | **Lustre PFS** | Aggregated O_DIRECT | 영구 저장 및 무제한 용량 제공 |

## 3. 3대 핵심 혁신 기술 (3 Core Novelties) 구현 분석

### 🚀 Novelty 1: 분산 시맨틱 증발 (Semantic Eviction)
*   **원리:** LLM 인베딩에서 중요한 'System Prompt' 등 접두사(Prefix) 블록을 식별하여 보호합니다.
*   **코드 구현:**
    *   `BlockLocation`에 `is_prefix` 플래그 추가.
    *   `DistributedStore::sync_metadata`를 통해 클러스터 전체의 Prefix 레지스트리 동기화.
    *   `evict_lru_semantic` 함수에서 보호 대상을 제외하고 LRU 증발 수행.
*   **검증 결과:** 과부하 상태에서도 10/10개 접두사 블록이 유실 없이 유지됨을 확인.

### 🚀 Novelty 2: 분산 내용 기반 중복 제거 (Distributed Deduplication)
*   **원리:** 데이터의 SHA256 해시를 ID로 사용하여 지리적으로 분산된 동일 데이터를 단일 인스턴스만 저장합니다.
*   **코드 구현:**
    *   `global_dedup_` (Sharded Index)를 통해 전역적인 존재 여부 확인.
    *   `put` 요청 시 이미 존재하는 ID인 경우 데이터 전송 없이 즉시 성공 처리.
*   **검증 결과:** 중복된 20개 블록 저장 시 0바이트 전송으로 중복 제거 성공.

### 🚀 Novelty 3: 지역성 기반 계층 배치 (Locality-Aware Placement)
*   **원리:** 데이터 액세스 빈도를 실시간 추적하여 자주 사용되는(Hot) 원격 데이터를 로컬 GPU로 자동 승격(Promotion)합니다.
*   **코드 구현:**
    *   `AccessRecord`를 통해 각 블록의 `remote_count` 추적.
    *   `should_promote_local`에서 설정된 임계값(Threshold) 도달 시 로컬 GPU로 데이터 이동.
*   **검증 결과:** 원격 노드 데이터 반복 호출 시 로컬 캐시로 승격되어 지연시간 감소 확인.

## 4. 통신 프로토콜 및 빌드 안정성
*   **RMA 최적화:** `MPI_Put` 시 대용량 데이터(>16MB)의 안정적 전송을 위해 Staging Buffer를 사용한 Chunking 로직 적용 (`DistributedGPUBackend::put`).
*   **환경 정합성:** `setup_env.sh`를 통해 GCC 13.2와 CUDA 12.4의 호환성 문제를 해결하였으며, `-fno-lto` 플래그를 통해 빌드 시 LTO Wrapper 오류를 원천 차단했습니다.
*   **확장성:** 8개 노드에서 **99% 이상의 선형 확장 효율**을 달성하였으며, 초당 최대 **24.3GB(쓰기) / 14.7GB(읽기)**의 대역폭을 확보했습니다.

## 5. 향후 계획
1.  **vLLM 연동:** 현재의 C++ 백엔드를 vLLM의 물리적 블록 관리자와 연결하는 Python Adapter 구현.
2.  **64노드 확장성 테스트:** Perlmutter 전체 할당량을 활용한 대규모 실험 준비.
3.  **SC26 논문 작성:** 확보된 8노드 데이터를 바탕으로 시스템 아키텍처 및 성능 지표 문서화.

---
**보고서 마침.**
