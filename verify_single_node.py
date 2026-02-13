import os
import sys
import time
import numpy as np
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

print(f"REPO_ROOT: {REPO_ROOT}")

try:
    import cascade_cpp
    import cascade
    print("✅ cascade_cpp and cascade wrapper imported")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def verify_cascade_core():
    print("\n--- Verifying Cascade Core (SHM) ---")
    cfg = cascade_cpp.CascadeConfig()
    cfg.shm_path = "/dev/shm/cascade_verify"
    cfg.shm_capacity_bytes = 1024 * 1024 * 1024  # 1GB
    cfg.use_gpu = False
    
    store = cascade_cpp.CascadeStore(cfg)
    print(f"Store initialized at {cfg.shm_path}")
    
    # 80MB block (typical size)
    size = 80 * 1024 * 1024
    data = np.random.randint(0, 256, size, dtype=np.uint8)
    block_id = "test_block_0"
    
    print(f"Writing {size/1024/1024:.1f} MB block...")
    t0 = time.perf_counter()
    store.put(block_id, data, False)
    t1 = time.perf_counter()
    print(f"Write throughput: {size/1e9/(t1-t0):.2f} GB/s")
    
    print("Reading block...")
    out = np.zeros(size, dtype=np.uint8)
    t0 = time.perf_counter()
    found, read_size = store.get(block_id, out)
    t1 = time.perf_counter()
    
    if not found:
        print("❌ Block NOT found!")
        return False
    
    print(f"Read throughput: {size/1e9/(t1-t0):.2f} GB/s")
    
    if np.array_equal(data, out):
        print("✅ Data integrity check PASSED")
    else:
        print("❌ Data integrity check FAILED")
        return False
        
    print(f"Stats: {store.get_stats()}")
    return True

def verify_lustre_backend():
    print("\n--- Verifying Lustre Backend ---")
    lustre_path = REPO_ROOT / "verify_lustre_store"
    lustre_path.mkdir(exist_ok=True)
    
    # We use direct Backend class to test individual tiers
    backend = cascade_cpp.LustreBackend(str(lustre_path))
    
    size = 10 * 1024 * 1024 # 10MB
    data = np.random.randint(0, 256, size, dtype=np.uint8)
    block_id = "lustre_test_0"
    
    print(f"Writing {size/1024/1024:.1f} MB to Lustre...")
    backend.put(block_id, data)
    
    print("Reading from Lustre...")
    out = np.zeros(size, dtype=np.uint8)
    found, read_size = backend.get(block_id, out)
    
    if found and np.array_equal(data, out):
        print("✅ Lustre backend check PASSED")
    else:
        print("❌ Lustre backend check FAILED")
        return False
    return True

if __name__ == "__main__":
    success = True
    success &= verify_cascade_core()
    success &= verify_lustre_backend()
    
    if success:
        print("\n✨ ALL SINGLE-NODE CHECKS PASSED ✨")
    else:
        print("\n🛑 SOME CHECKS FAILED")
        sys.exit(1)
