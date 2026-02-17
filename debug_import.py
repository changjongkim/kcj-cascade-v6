import sys
import os
print("--- Importing cascade_cpp ---")
try:
    import cascade_cpp
    print(f"SUCCESS: {cascade_cpp.__file__}")
    if hasattr(cascade_cpp, 'DistributedStore'):
        print(f"DistributedStore.clear: {hasattr(cascade_cpp.DistributedStore, 'clear')}")
        print(f"DistributedStore.flush: {hasattr(cascade_cpp.DistributedStore, 'flush')}")
except Exception as e:
    print(f"FAIL: {e}")
