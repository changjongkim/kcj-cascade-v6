import sys
sys.path.insert(0, "/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache")

try:
    from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
    print("SUCCESS importing LocalDiskBackend")
except Exception as e:
    print(f"FAILED to import LocalDiskBackend: {e}")
    import traceback
    traceback.print_exc()
