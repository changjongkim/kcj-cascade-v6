# Cascade: HPC 스케일 LLM 추론을 위한 KV 캐시 스토리지 시스템

> **SC'26 논문** | NERSC Perlmutter | A100 GPU | Slingshot-11

---

## 🎯 문제 정의

LLM 추론은 **메모리 바운드**: KV 캐시 로딩이 병목입니다.

| 시스템 | 유형 | 한계점 |
|--------|------|--------|
| **vLLM** | GPU 메모리 | GPU당 40GB 제한 |
| **LMCache** | 파일 기반 | 싱글노드 전용 |
| **PDC** | 객체 스토리지 | fsync 오버헤드 |
| **Redis** | 인메모리 KV | 네트워크 직렬화 |
| **HDF5** | 과학 데이터 | 압축 CPU 병목 |

---

## 🚀 Cascade 아키텍처

\`\`\`
Tier 1: GPU HBM     ─── 1555 GB/s, 40GB × 4 = 160GB/노드
   ↓ evict (async)
Tier 2: 로컬 SHM    ─── 204 GB/s, /dev/shm 256GB/노드
   ↓ MPI 전송
Tier 3: 원격 SHM    ─── 22.8 GB/s, Slingshot-11
   ↓ async prefetch
Tier 4: Lustre PFS  ─── 7.8 TB/s aggregate, \$SCRATCH
\`\`\`

---

## 📦 디렉토리 구조

\`\`\`
/pscratch/sd/s/sgkim/Skim-cascade/
├── cascade_Code/           # Cascade 구현
│   ├── cpp/               # C++ 코어 (mmap, MPI, 분산)
│   └── src/cascade/       # Python 래퍼
├── third_party/            # 비교 시스템들 (실제 설치)
│   ├── LMCache/           # LMCache (torch 필요)
│   ├── pdc/               # PDC (C, MPI)
│   ├── redis/             # Redis (C)
│   └── vllm/              # vLLM 참조용
├── benchmark/              # 벤치마크
└── paper/                  # SC'26 논문
\`\`\`

---

# 🔧 설치 및 사용 가이드

## 1. Cascade (우리 시스템)

### 1.1 C++ MPI 분산 버전 빌드

\`\`\`bash
cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp
mkdir build_mpi && cd build_mpi

srun -A m1248_g -C gpu -q debug -n 1 -c 64 --gpus=4 -t 00:10:00 bash -c '
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DPERLMUTTER=ON
make -j32 distributed_bench
'
\`\`\`

### 1.2 실행 (멀티노드)

\`\`\`bash
srun -A m1248_g -C gpu -q debug -N 4 -n 4 --gpus-per-node=4 \\
    --export=ALL,MPICH_GPU_SUPPORT_ENABLED=1 \\
    ./distributed_bench --blocks 1000 --block-size 10
\`\`\`

---

## 2. LMCache (Baseline)

### 위치: third_party/LMCache/

\`\`\`bash
# GPU 노드에서 (torch 필요)
srun -A m1248_g -C gpu -N 1 --gpus=4 -t 00:30:00 bash -c '
module load python cudatoolkit
python -c "
import sys
sys.path.insert(0, \"/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache\")
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
print(\"LMCache OK\")
"
'
\`\`\`

---

## 3. PDC (Proactive Data Containers)

### 위치: third_party/pdc/install/

\`\`\`bash
export PDC_DIR=/pscratch/sd/s/sgkim/Skim-cascade/third_party/pdc/install
export PATH=\$PDC_DIR/bin:\$PATH
export LD_LIBRARY_PATH=\$PDC_DIR/lib:\$LD_LIBRARY_PATH

# 서버 시작
pdc_server &
\`\`\`

---

## 4. Redis

### 위치: third_party/redis/src/

\`\`\`bash
# GPU 노드에서 (libcudart 필요)
srun -A m1248_g -C gpu -N 1 --gpus=1 -t 00:30:00 bash -c '
module load cudatoolkit
/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server --daemonize yes
'
\`\`\`

---

## 5. HDF5 (h5py)

\`\`\`bash
module load python
pip install h5py --user
python -c "import h5py; print(h5py.__version__)"
\`\`\`

---

# 📊 벤치마크 실행

## Hot vs Cold Read

- **Hot**: 데이터가 SHM 또는 page cache에 있을 때
- **Cold**: posix_fadvise(DONTNEED)로 cache 비운 후 Lustre 직접 읽기

\`\`\`python
import os, ctypes

def drop_page_cache(path):
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    libc = ctypes.CDLL("libc.so.6")
    libc.posix_fadvise(fd, 0, size, 4)  # POSIX_FADV_DONTNEED
    os.close(fd)
\`\`\`

---

# 📋 실험 환경 (Perlmutter)

| 구성요소 | 사양 |
|---------|------|
| **GPU** | NVIDIA A100-40GB × 4 = 160GB HBM/노드 |
| **CPU** | AMD EPYC 7763 (64 cores) |
| **DRAM** | 256GB DDR4/노드 |
| **SHM** | /dev/shm: ~428GB |
| **인터커넥트** | Slingshot-11 (200 Gb/s × 4 NIC) |
| **스토리지** | Lustre \$SCRATCH (44PB, 7.8 TB/s) |

---

# ⚠️ 연구 윤리

> **절대 하면 안되는 것**:
> - 가짜 벤치마크 (단순 파일 I/O로 시뮬레이션)
> - 실제 시스템 사용하지 않고 "LMCache" 등으로 레이블링

> **반드시 해야 하는 것**:
> - third_party/의 실제 구현 사용
> - Job ID와 함께 재현 가능한 결과 보고

---

# 📈 벤치마크 결과

> ⚠️ **TODO**: 실제 벤치마크 결과 수집 중
> 아래 테스트를 실제 시스템으로 실행해야 합니다.

| 테스트 | 시스템 | 상태 |
|--------|--------|------|
| Hot Read | Cascade, LMCache, PDC, Redis, HDF5 | 미완료 |
| Cold Read | Cascade, LMCache, PDC, Redis, HDF5 | 미완료 |

---

## 📖 라이센스

SC'26 논문 제출용 연구 프로젝트
