#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH -J real_5sys
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_5sys_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_5sys_%j.err

###############################################################################
# REAL 5-SYSTEM BENCHMARK - SC'26
# 🚨 RESEARCH INTEGRITY: Uses actual third_party implementations
###############################################################################

set -e
echo "=========================================="
echo "REAL 5-SYSTEM BENCHMARK"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Date: $(date -Iseconds)"
echo "=========================================="

# Environment setup
module load python cudatoolkit
export PYTHONPATH="/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache:$PYTHONPATH"
export LD_LIBRARY_PATH="/pscratch/sd/s/sgkim/Skim-cascade/third_party/pdc/install/lib:$LD_LIBRARY_PATH"

cd /pscratch/sd/s/sgkim/Skim-cascade

# Test parameters
NUM_BLOCKS=100
BLOCK_SIZE_MB=10
TOTAL_DATA_MB=$((NUM_BLOCKS * BLOCK_SIZE_MB))

RESULT_FILE="benchmark/results/real_5sys_${SLURM_JOB_ID}.json"

echo ""
echo "=== Test Configuration ==="
echo "Blocks: $NUM_BLOCKS"
echo "Block Size: ${BLOCK_SIZE_MB} MB"
echo "Total Data per rank: ${TOTAL_DATA_MB} MB"
echo ""

###############################################################################
# 1. CASCADE C++ MPI Distributed Benchmark
###############################################################################
echo "=== [1/5] Cascade C++ MPI Distributed ==="

CASCADE_BIN="cascade_Code/cpp/build_mpi/distributed_bench"
if [ -f "$CASCADE_BIN" ]; then
    echo "Running Cascade distributed benchmark..."
    
    srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 \
        --export=ALL,MPICH_GPU_SUPPORT_ENABLED=1 \
        $CASCADE_BIN --blocks $NUM_BLOCKS --block-size $BLOCK_SIZE_MB 2>&1 | tee /tmp/cascade_result.txt
    
    CASCADE_WRITE=$(grep -i "write.*gb/s\|write.*throughput" /tmp/cascade_result.txt | tail -1 || echo "N/A")
    CASCADE_READ=$(grep -i "read.*gb/s\|read.*throughput" /tmp/cascade_result.txt | tail -1 || echo "N/A")
    echo "Cascade Write: $CASCADE_WRITE"
    echo "Cascade Read: $CASCADE_READ"
else
    echo "ERROR: Cascade binary not found at $CASCADE_BIN"
    CASCADE_WRITE="ERROR"
    CASCADE_READ="ERROR"
fi

###############################################################################
# 2. LMCACHE - Real Implementation
###############################################################################
echo ""
echo "=== [2/5] LMCache (Real third_party) ==="

python3 << 'LMCACHE_BENCH'
import sys
import time
import os
import numpy as np
import json

# Add LMCache to path
sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')

try:
    # Try to import real LMCache
    from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
    print("✅ LMCache LocalDiskBackend imported successfully")
    
    # Create test data
    num_blocks = 100
    block_size = 10 * 1024 * 1024  # 10MB
    data = np.random.randint(0, 255, size=block_size, dtype=np.uint8).tobytes()
    
    # Prepare storage path
    storage_path = f"/tmp/lmcache_bench_{os.getpid()}"
    os.makedirs(storage_path, exist_ok=True)
    
    # Initialize LMCache backend
    backend = LocalDiskBackend(path=storage_path, max_size=100*1024**3)
    
    # Write benchmark
    print(f"Writing {num_blocks} blocks of {block_size/1024/1024:.1f}MB each...")
    start = time.perf_counter()
    for i in range(num_blocks):
        key = f"block_{i:06d}"
        backend.put(key.encode(), data)
    write_time = time.perf_counter() - start
    total_bytes = num_blocks * block_size
    write_gbps = (total_bytes / 1e9) / write_time
    print(f"LMCache Write: {write_gbps:.2f} GB/s ({write_time:.3f}s)")
    
    # Hot Read benchmark (data in page cache)
    print("Hot read (page cache)...")
    start = time.perf_counter()
    for i in range(num_blocks):
        key = f"block_{i:06d}"
        result = backend.get(key.encode())
    hot_time = time.perf_counter() - start
    hot_gbps = (total_bytes / 1e9) / hot_time
    print(f"LMCache Hot Read: {hot_gbps:.2f} GB/s ({hot_time:.3f}s)")
    
    # Save results
    with open('/tmp/lmcache_result.json', 'w') as f:
        json.dump({
            'system': 'LMCache',
            'implementation': 'third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py',
            'write_gbps': write_gbps,
            'hot_read_gbps': hot_gbps,
            'num_blocks': num_blocks,
            'block_size_mb': block_size / 1024 / 1024
        }, f, indent=2)
    
    # Cleanup
    import shutil
    shutil.rmtree(storage_path, ignore_errors=True)
    
except ImportError as e:
    print(f"❌ LMCache import failed: {e}")
    print("Note: LMCache requires torch which needs GPU node")
    with open('/tmp/lmcache_result.json', 'w') as f:
        json.dump({'system': 'LMCache', 'error': str(e)}, f)
except Exception as e:
    print(f"❌ LMCache error: {e}")
    with open('/tmp/lmcache_result.json', 'w') as f:
        json.dump({'system': 'LMCache', 'error': str(e)}, f)
LMCACHE_BENCH

###############################################################################
# 3. HDF5 - Real h5py Implementation
###############################################################################
echo ""
echo "=== [3/5] HDF5 (Real h5py) ==="

python3 << 'HDF5_BENCH'
import sys
import time
import os
import numpy as np
import json

try:
    import h5py
    print("✅ h5py imported successfully")
    
    num_blocks = 100
    block_size = 10 * 1024 * 1024  # 10MB
    
    # Create test data
    data = np.random.randint(0, 255, size=(num_blocks, block_size // 4), dtype=np.int32)
    
    h5_path = f"/tmp/hdf5_bench_{os.getpid()}.h5"
    
    # Write benchmark (with compression)
    print(f"Writing {num_blocks} blocks with gzip compression...")
    start = time.perf_counter()
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('kv_cache', data=data, compression='gzip', compression_opts=1)
    write_time = time.perf_counter() - start
    total_bytes = data.nbytes
    write_gbps = (total_bytes / 1e9) / write_time
    print(f"HDF5 Write (gzip): {write_gbps:.2f} GB/s ({write_time:.3f}s)")
    
    # Hot Read benchmark
    print("Hot read...")
    start = time.perf_counter()
    with h5py.File(h5_path, 'r') as f:
        result = f['kv_cache'][:]
    hot_time = time.perf_counter() - start
    hot_gbps = (total_bytes / 1e9) / hot_time
    print(f"HDF5 Hot Read: {hot_gbps:.2f} GB/s ({hot_time:.3f}s)")
    
    # Save results
    with open('/tmp/hdf5_result.json', 'w') as f:
        json.dump({
            'system': 'HDF5',
            'implementation': 'h5py with gzip compression',
            'write_gbps': write_gbps,
            'hot_read_gbps': hot_gbps,
            'num_blocks': num_blocks,
            'block_size_mb': block_size / 1024 / 1024
        }, f, indent=2)
    
    os.remove(h5_path)
    
except ImportError as e:
    print(f"❌ h5py not installed: {e}")
    with open('/tmp/hdf5_result.json', 'w') as f:
        json.dump({'system': 'HDF5', 'error': str(e)}, f)
except Exception as e:
    print(f"❌ HDF5 error: {e}")
    with open('/tmp/hdf5_result.json', 'w') as f:
        json.dump({'system': 'HDF5', 'error': str(e)}, f)
HDF5_BENCH

###############################################################################
# 4. Redis - Real Implementation
###############################################################################
echo ""
echo "=== [4/5] Redis (Real third_party) ==="

REDIS_SERVER="/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server"
REDIS_CLI="/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-cli"

if [ -f "$REDIS_SERVER" ]; then
    echo "Starting Redis server..."
    $REDIS_SERVER --port 6379 --daemonize yes --dir /tmp 2>/dev/null || echo "Redis may already be running"
    sleep 2
    
    python3 << 'REDIS_BENCH'
import sys
import time
import os
import numpy as np
import json

try:
    # Try to import redis-py
    sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/python_pkgs_py312/lib')
    import redis
    print("✅ redis-py imported successfully")
    
    client = redis.Redis(host='localhost', port=6379)
    client.ping()
    print("✅ Connected to Redis server")
    
    num_blocks = 100
    block_size = 10 * 1024 * 1024  # 10MB
    data = np.random.randint(0, 255, size=block_size, dtype=np.uint8).tobytes()
    
    # Write benchmark
    print(f"Writing {num_blocks} blocks of {block_size/1024/1024:.1f}MB each...")
    start = time.perf_counter()
    pipe = client.pipeline()
    for i in range(num_blocks):
        pipe.set(f"block_{i:06d}", data)
    pipe.execute()
    write_time = time.perf_counter() - start
    total_bytes = num_blocks * block_size
    write_gbps = (total_bytes / 1e9) / write_time
    print(f"Redis Write: {write_gbps:.2f} GB/s ({write_time:.3f}s)")
    
    # Read benchmark
    print("Reading blocks...")
    start = time.perf_counter()
    pipe = client.pipeline()
    for i in range(num_blocks):
        pipe.get(f"block_{i:06d}")
    results = pipe.execute()
    read_time = time.perf_counter() - start
    read_gbps = (total_bytes / 1e9) / read_time
    print(f"Redis Read: {read_gbps:.2f} GB/s ({read_time:.3f}s)")
    
    # Cleanup
    client.flushall()
    
    with open('/tmp/redis_result.json', 'w') as f:
        json.dump({
            'system': 'Redis',
            'implementation': 'third_party/redis/src/redis-server + redis-py',
            'write_gbps': write_gbps,
            'hot_read_gbps': read_gbps,
            'num_blocks': num_blocks,
            'block_size_mb': block_size / 1024 / 1024
        }, f, indent=2)

except ImportError as e:
    print(f"❌ redis-py import failed: {e}")
    with open('/tmp/redis_result.json', 'w') as f:
        json.dump({'system': 'Redis', 'error': str(e)}, f)
except Exception as e:
    print(f"❌ Redis error: {e}")
    with open('/tmp/redis_result.json', 'w') as f:
        json.dump({'system': 'Redis', 'error': str(e)}, f)
REDIS_BENCH

    # Stop Redis
    $REDIS_CLI shutdown 2>/dev/null || true
else
    echo "Redis binary not found"
fi

###############################################################################
# 5. PDC - Proactive Data Containers
###############################################################################
echo ""
echo "=== [5/5] PDC (Proactive Data Containers) ==="

PDC_SERVER="/pscratch/sd/s/sgkim/Skim-cascade/third_party/pdc/install/bin/pdc_server"
if [ -f "$PDC_SERVER" ]; then
    echo "PDC server binary exists"
    echo "Note: PDC requires C API integration for proper benchmarking"
    echo "Skipping PDC for this run - needs dedicated C benchmark"
else
    echo "PDC server not found"
fi

###############################################################################
# Aggregate Results
###############################################################################
echo ""
echo "=========================================="
echo "AGGREGATING RESULTS"
echo "=========================================="

python3 << AGGREGATE
import json
import os
from datetime import datetime

results = {
    'job_id': os.environ.get('SLURM_JOB_ID', 'unknown'),
    'timestamp': datetime.now().isoformat(),
    'nodes': int(os.environ.get('SLURM_NNODES', 1)),
    'config': {
        'num_blocks': 100,
        'block_size_mb': 10,
        'total_data_mb': 1000
    },
    'systems': {}
}

# Load individual results
for sys_name in ['lmcache', 'hdf5', 'redis']:
    result_file = f'/tmp/{sys_name}_result.json'
    if os.path.exists(result_file):
        with open(result_file) as f:
            results['systems'][sys_name] = json.load(f)
    else:
        results['systems'][sys_name] = {'error': 'No result file'}

# Read Cascade result from stdout capture
cascade_file = '/tmp/cascade_result.txt'
if os.path.exists(cascade_file):
    with open(cascade_file) as f:
        cascade_output = f.read()
    results['systems']['cascade'] = {
        'implementation': 'cascade_Code/cpp/build_mpi/distributed_bench',
        'output': cascade_output[:2000]  # First 2000 chars
    }

# Save aggregated results
result_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/real_5sys_{os.environ.get('SLURM_JOB_ID', 'unknown')}.json"
with open(result_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {result_path}")
print(json.dumps(results, indent=2))
AGGREGATE

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
