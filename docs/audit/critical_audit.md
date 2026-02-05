# Cascade 코드 비판적 감사 보고서 (Evidence-Based Critical Audit)

이 보고서는 Cascade 코드베이스에 존재하는 심각한 결함들을 실제 코드 증거(Proof)를 통해 입증합니다. 현재 코드는 README의 주장과 달리 실제 환경에서 작동하지 않거나 성능을 조작하고 있습니다.

---

## 1. Lustre 스토리지 성능 조작 (Fake Benchmark)
**주장**: 비동기 I/O (io_uring)를 사용하여 고속 디스크 쓰기 수행.
**실제**: 표준 `write()`를 사용하여 **RAM(페이지 캐시)**에 기록. 디스크 속도가 아닌 메모리 복사 속도를 측정함.

### [Evidence] `src/cascade_core.cpp`
```cpp
// 실제 코드: O_DIRECT 플래그 누락 & 동기적 write 사용
bool LustreBackend::put(const BlockId& id, const uint8_t* data, size_t size) {
    // 1. O_DIRECT(디스크 직렬 쓰기) 플래그가 없음. -> OS가 RAM(Page Cache)에 씀.
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    
    // 2. 동기(Synchronous) 함수 호출. io_uring 아님. 
    // 하지만 RAM에 쓰기 때문에 매우 빠른 것처럼 보임 (Fake Latency).
    write(fd, data, size); 
    close(fd);
    return true;
}
```
**결과**: 디스크가 아무리 느려도 벤치마크는 수 GB/s가 나옴(RAM 속도). 실제 데이터는 아직 디스크에 안 써짐.

---

## 2. 분산 처리(MPI) 데이터 증발 (Data Loss)
**주장**: 원격 노드의 메모리를 활용하여 용량 확장 (Tier 3).
**실제**: 데이터를 원격 노드의 '임시 버퍼'로 던지기만 하고, **받는 쪽에서 저장하는 로직이 없음.**

### [Evidence] `src/distributed_backend.cpp`
```cpp
bool DistributedGPUBackend::put(const BlockId& id, const uint8_t* data, size_t size) {
    // ... (데이터 준비) ...

    // 1. MPI_Put으로 원격 노드의 'staging_buffer'(임시 공간)에 복사
    MPI_Put(pinned_[0], chunk, MPI_BYTE, target_node, 0, chunk, MPI_BYTE, window_);
    
    // 2. 중요: 원격 노드에게 "데이터 왔으니 가져가라"고 알리는 신호(Signal)가 없음.
    // 3. 치명적 결함: 저장 로직 부재를 인정하는 TODO 주석 방치.
    // TODO: Need a protocol for remote node to store from staging to GPU
    
    return true; // 저장이 안 됐는데 성공(true) 리턴.
}
```
**결과**: 원격 노드로 보낸 데이터는 다음 요청이 오면 덮어씌워져 사라짐. 나중에 `get` 하면 찾을 수 없음.

---

## 3. 메모리 누수 및 지속 불가능한 구조 (Memory Leak)
**주장**: HPC 스케일의 고성능 캐시 시스템.
**실제**: 메모리를 해제하는 기능이 없어, 캐시가 차면 시스템이 죽음 (OOM).

### [Evidence] `src/gpu_backend.cu`
```cpp
class GPUMemoryPool {
    // ...
    // 1. 할당(Alloc): 오프셋만 증가시킴 (Bump Allocation)
    void* alloc(size_t size) {
        size_t offset = current_offset_.fetch_add(aligned_size);
        if (offset > capacity_) return nullptr;
        return base_ptr + offset;
    }

    // 2. 해제(Free): 개별 반환 함수 자체가 아예 없음.
    // 3. Reset: 오직 '전체 초기화'만 가능.
    void reset() {
        current_offset_ = 0;
    }
};
```
**결과**: 실제 서비스(vLLM 등)는 끊임없이 블록을 생성/삭제하는데, 삭제가 안 되니 수 분 내에 메모리가 가득 차서 서버가 멈춤.

---

## 4. SHA256 병목 (Performance Bottleneck)
**주장**: 25GB/s 이상의 초고속 처리량 (Zero-Copy).
**실제**: 모든 데이터에 대해 CPU로 SHA256을 계산하느라 1GB/s 넘기기도 힘듦.

### [Evidence] `src/cascade_core.cpp`
```cpp
BlockId compute_block_id(const uint8_t* data, size_t size) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    // 1. 동기적(Synchronous) 해시 계산.
    // 4KB 블록마다 호출됨. CPU가 100%를 찍으며 병목 발생.
    SHA256(data, size, hash); 
    // 네트워크가 100GB/s여도 CPU 해시 속도가 500MB/s면 전체 성능은 500MB/s.
    return bytes_to_hex(hash);
}
```

---

## 5. 가짜 Lock-Free 및 빌드 설정 기만
**주장**: Lock-Free Indexing, Async I/O (io_uring).
**실제**: Mutex 사용, 기본 빌드에서 Async 꺼짐.

### [Evidence 1] `include/cascade.hpp` (Lock-Free 거짓말)
```cpp
template<typename V>
class ShardedIndex {
    struct Shard {
        // "Lock-Free"라면서 Mutex를 쓰고 있음. 경합 발생 시 성능 저하.
        mutable std::shared_mutex mutex; 
        std::unordered_map<BlockId, V> data;
    };
};
```

### [Evidence 2] `CMakeLists.txt` (Async I/O 거짓말)
```cmake
# 사용자가 명시적으로 켜지 않으면 기본값은 OFF (동기 I/O)
option(USE_IOURING "Enable io_uring for async I/O" OFF)
```

---

## 종합 결론

이 코드는 **"논문 제출용 벤치마크 수치 생성기"**입니다.
1.  **RAM을 디스크인 척 측정**하여 스토리지 성능을 속임 (Lustre).
2.  **저장하지 않고 성공 처리**하여 네트워크 성능을 속임 (Distributed).
3.  **메모리 해제를 포기**하여 구현 복잡도를 회피함 (Memory Pool).

실제 서비스에 투입할 경우 **데이터 유실(Data Loss)**과 **시스템 멈춤(OOM)**이 즉시 발생합니다.
