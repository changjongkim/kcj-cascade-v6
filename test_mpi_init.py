import os
import sys

# Path setup
default_build = 'cascade_Code/cpp/build_cascade_cpp'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.getcwd(), build_dir)
sys.path.insert(0, build_dir)

try:
    import cascade_cpp
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    
    print(f"Before Cascade: MPI.Is_initialized() = {MPI.Is_initialized()}")
    
    cfg = cascade_cpp.DistributedConfig()
    cfg.gpu_capacity_per_device = 1 * 1024**3
    cfg.dram_capacity = 1 * 1024**3
    store = cascade_cpp.DistributedStore(cfg)
    
    print(f"After Cascade: MPI.Is_initialized() = {MPI.Is_initialized()}")
    
    if not MPI.Is_initialized():
        print("Initializing MPI via mpi4py...")
        MPI.Init()
        print("MPI Initialized via mpi4py.")
    
    MPI.COMM_WORLD.Barrier()
    print("Barrier passed.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
