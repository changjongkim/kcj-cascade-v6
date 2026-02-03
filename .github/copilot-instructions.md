# Cascade: HPC-Scale KV Cache Storage System

## 🚨🚨🚨 연구 윤리 - 가장 중요 🚨🚨🚨

### 절대 하면 안되는 것 (연구부정)

1. **가짜 벤치마크 금지**
   - 단순 Python 파일 I/O를 "LMCache", "PDC", "Redis"로 레이블링 금지
   - 실제 third_party 구현을 사용하지 않으면 연구부정

2. **시뮬레이션 금지**
   - 모든 비교 시스템은 third_party/의 실제 코드 사용 필수
   - 성능 추정/예측 결과를 실험 결과로 제시 금지

3. **선별적 보고 금지**
   - 유리한 결과만 선택 보고 금지
   - 모든 실험 조건과 Job ID 명시 필수

### 반드시 해야 하는 것

1. 모든 벤치마크에서 실제 third_party 시스템 사용
2. 실험 환경 상세 명시 (노드 수, 데이터 크기, Hot/Cold 상태)
3. Job ID로 재현 가능성 보장
4. Cold read 테스트 시 posix_fadvise(DONTNEED) 사용하여 page cache 비우기

---

## Project Overview

Cascade: LLM 추론을 위한 4계층 KV 캐시 스토리지 시스템
- Target: SC'26 논문
- Platform: NERSC Perlmutter (A100 GPU, Slingshot-11)

---

## 실제 third_party 시스템 사용법

### 1. LMCache

**위치**: `/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache/`

**의존성**: torch (GPU 노드 필수)

**실제 코드 import**:
```python
import sys
sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')

# 실제 LMCache storage backend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
```

**주의**: GPU 노드에서만 실행 가능 (torch 필요)

### 2. PDC (Proactive Data Containers)

**위치**: `/pscratch/sd/s/sgkim/Skim-cascade/third_party/pdc/`

**설치 확인**:
- `install/bin/pdc_server` - PDC 서버 실행파일
- `install/bin/close_server` - 서버 종료

**사용법**:
```bash
export PDC_DIR=/pscratch/sd/s/sgkim/Skim-cascade/third_party/pdc/install
export PATH=$PDC_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PDC_DIR/lib:$LD_LIBRARY_PATH

# 서버 시작
pdc_server &
```

**C API 사용**:
```c
#include "pdc.h"
pdcid_t pdc = PDCinit("pdc");
// ... PDC operations
PDCclose(pdc);
```

### 3. Redis

**위치**: `/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/`

**의존성**: libcudart.so.12 (GPU 노드 필수)

**실행** (GPU 노드에서):
```bash
module load cudatoolkit
/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server \
    --port 6379 --daemonize yes
```

**Python 클라이언트**:
```python
import redis
client = redis.Redis(host='localhost', port=6379)
```

### 4. HDF5 (h5py)

**설치**:
```bash
module load python
pip install h5py --user
```

**사용**:
```python
import h5py
import numpy as np

with h5py.File('data.h5', 'w') as f:
    f.create_dataset('kv_cache', data=array, compression='gzip')
```

---

## Cascade 사용법

### C++ MPI 분산 버전

**빌드** (GPU 노드에서):
```bash
cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp
mkdir build_mpi && cd build_mpi

srun -A m1248_g -C gpu -q debug -n 1 -c 64 --gpus=4 -t 00:10:00 bash -c '
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DPERLMUTTER=ON
make -j32 distributed_bench
'
```

**실행** (멀티노드):
```bash
srun -A m1248_g -C gpu -q debug -N 4 -n 4 --gpus-per-node=4 \
    --export=ALL,MPICH_GPU_SUPPORT_ENABLED=1 \
    ./distributed_bench --blocks 1000 --block-size 10
```

### 주요 파일

| 파일 | 설명 |
|------|------|
| `cascade_Code/cpp/include/cascade_distributed.hpp` | 분산 백엔드 헤더 |
| `cascade_Code/cpp/src/distributed_backend.cpp` | MPI RMA 구현 |
| `cascade_Code/cpp/src/distributed_benchmark.cpp` | 멀티노드 벤치마크 |

---

## Hot vs Cold Read 벤치마크

### Hot Read
데이터가 SHM 또는 OS page cache에 있을 때 측정

### Cold Read
page cache를 비운 후 Lustre에서 직접 읽기

```python
import os
import ctypes

def drop_page_cache(path):
    """파일의 page cache를 비웁니다 (Cold read 테스트용)"""
    fd = os.open(path, os.O_RDONLY)
    file_size = os.fstat(fd).st_size
    libc = ctypes.CDLL("libc.so.6")
    # POSIX_FADV_DONTNEED = 4
    libc.posix_fadvise(fd, 0, file_size, 4)
    os.close(fd)
```

---

## 실험 환경 (Perlmutter)

| 구성요소 | 사양 |
|---------|------|
| GPU | NVIDIA A100-40GB × 4 = 160GB HBM/노드 |
| CPU | AMD EPYC 7763 (64 cores) |
| DRAM | 256GB DDR4/노드 |
| SHM | /dev/shm: ~428GB 사용 가능 |
| 인터커넥트 | Slingshot-11 (200 Gb/s × 4 NIC) |
| 스토리지 | Lustre $SCRATCH (44PB, 7.8 TB/s aggregate) |

---

## 벤치마크 결과 보고 형식

모든 결과는 다음 정보를 포함해야 합니다:

```json
{
  "job_id": "SLURM_JOB_ID",
  "timestamp": "2026-02-02T16:51:33",
  "environment": {
    "nodes": 4,
    "ranks_per_node": 4,
    "total_ranks": 16,
    "gpus_per_node": 4,
    "total_gpus": 16
  },
  "test_config": {
    "block_size_mb": 10,
    "num_blocks": 100,
    "total_data_gb": 16,
    "hot_or_cold": "cold",
    "page_cache_dropped": true
  },
  "systems_tested": {
    "Cascade": "cascade_Code/cpp/build_mpi/distributed_bench",
    "LMCache": "third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py",
    "PDC": "third_party/pdc/install/bin/pdc_server",
    "Redis": "third_party/redis/src/redis-server",
    "HDF5": "h5py library"
  },
  "results": {
    "Cascade": {"write_gbps": 0.0, "hot_gbps": 0.0, "cold_gbps": 0.0},
    "LMCache": {"write_gbps": 0.0, "hot_gbps": 0.0, "cold_gbps": 0.0}
  }
}
```

---

## 절대 하면 안되는 코드 패턴

```python
# ❌ 이런 코드는 연구부정입니다!
class LMCacheStore:
    """가짜 LMCache - 단순 파일 I/O"""
    def put(self, block_id, data):
        with open(f"{block_id}.bin", 'wb') as f:
            f.write(data)  # 이건 LMCache가 아닙니다!
```

```python
# ✅ 올바른 방법
import sys
sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

class LMCacheAdapter:
    def __init__(self):
        self.backend = LocalDiskBackend(path="/tmp/lmcache", max_size=100*1024**3)
    
    def put(self, block_id, data):
        self.backend.put(block_id, data)  # 실제 LMCache 코드 사용
```

---

## 디렉토리 구조

```
/pscratch/sd/s/sgkim/Skim-cascade/
├── cascade_Code/
│   ├── cpp/                    # C++ 코어 구현
│   │   ├── include/            # 헤더 파일
│   │   ├── src/                # 소스 파일
│   │   └── build_mpi/          # 빌드 산출물
│   └── src/cascade/            # Python 래퍼
├── third_party/                # 비교 시스템 (실제 설치)
│   ├── LMCache/                # torch 의존
│   ├── pdc/                    # C/MPI
│   ├── redis/                  # C, libcudart 의존
│   └── vllm/                   # 참조용
├── benchmark/
│   ├── adapters/               # 시스템 어댑터 (실제 구현 래핑)
│   ├── scripts/                # SLURM 스크립트
│   └── results/                # 결과 JSON (Job ID 포함)
└── paper/                      # SC'26 논문
```
