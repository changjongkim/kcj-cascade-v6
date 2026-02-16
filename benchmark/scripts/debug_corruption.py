
import os
import sys
import numpy as np
from pathlib import Path

# Path setup for Cascade library
default_build = '../../cascade_Code/cpp/build_cascade_cpp'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), build_dir)
sys.path.insert(0, build_dir)

import cascade_cpp

def test_corruption():
    print("Creating a test dictionary...")
    test_dict = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
    sample_key = "key1"
    
    print(f"Before init: {sample_key} in test_dict? {sample_key in test_dict}")
    print(f"Before init: value={test_dict[sample_key]}")
    
    print("Initializing DistributedStore with 10GB DRAM...")
    cfg = cascade_cpp.DistributedConfig()
    cfg.dram_capacity = 10 * 1024**3
    cfg.gpu_capacity_per_device = 1 * 1024**3
    cfg.num_gpus_per_node = 1
    
    store = cascade_cpp.DistributedStore(cfg)
    
    print(f"After init: {sample_key} in test_dict? {sample_key in test_dict}")
    try:
        print(f"After init: value={test_dict[sample_key]}")
    except KeyError:
        print("!!! CORRUPTION DETECTED: KeyError occurred after init!")
        print(f"Dict keys: {list(test_dict.keys())}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_corruption()
