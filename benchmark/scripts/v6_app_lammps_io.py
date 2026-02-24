import os
import sys
import time
import argparse
import numpy as np

rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))

# Path to built cascade_cpp.so
sys.path.append("/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build")

try:
    import cascade_cpp
except ImportError:
    if rank == 0:
        print("Error: Could not import cascade_cpp. Please build it first.")
    sys.exit(1)

# ============================================================
# Cascade Global Store (Ensemble style initialization)
# ============================================================
GLOBAL_STORE = None
def init_global_store():
    global GLOBAL_STORE
    if GLOBAL_STORE is None:
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 160 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        GLOBAL_STORE = cascade_cpp.DistributedStore(cfg)
    return GLOBAL_STORE

def mpi_barrier():
    if GLOBAL_STORE:
        GLOBAL_STORE.barrier()
    else:
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()
        except ImportError:
            pass

def print_rank0(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

class BaseStore:
    def put(self, key, data): raise NotImplementedError()
    def get(self, key, out): raise NotImplementedError()
    def barrier(self): mpi_barrier()
    def cleanup(self): pass

class CascadeAdapter(BaseStore):
    def __init__(self, tag):
        self.store = GLOBAL_STORE
        self.tag = tag
        self.bytes_put = 0
        mpi_barrier()

    def put(self, key, data):
        self.store.put(key, data, False)
        self.bytes_put += len(data)

    def get(self, key, out):
        return self.store.get(key, out)

    def storage_used(self):
        stats = self.store.get_stats()
        return self.bytes_put - stats.dedup_bytes_saved

    def sync(self):
        self.store.sync_metadata()
        self.barrier()
        self.store.sync_metadata()
        self.barrier()

import h5py
from pathlib import Path

class HDF5Adapter(BaseStore):
    def __init__(self, tag, file_per_rank=True):
        self.base_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_lammps_{tag}")
        if rank == 0:
            if self.base_dir.exists():
                import shutil
                shutil.rmtree(self.base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()
        # file-per-rank for faster writes without MPI-IO locking issues
        self.filepath = self.base_dir / f"rank_{rank}.h5"
        self.file = h5py.File(self.filepath, 'w')
        self.read_mode = False

    def put(self, key, data):
        if self.read_mode:
            self.file.close()
            self.file = h5py.File(self.filepath, 'a')
            self.read_mode = False
        if key in self.file:
            del self.file[key]
        self.file.create_dataset(key, data=data)
        self.file.flush()

    def open_for_read(self):
        self.file.close()
        self.file = h5py.File(self.filepath, 'r')
        self.read_mode = True

    def get(self, key, out):
        if not self.read_mode:
            self.open_for_read()
        if key in self.file:
            # For 1D proxy logic
            out[:] = self.file[key][:]
            return True
        return False

    def storage_used(self):
        if self.filepath.exists():
            return self.filepath.stat().st_size
        return 0

    def cleanup(self):
        mpi_barrier()
        if rank == 0 and self.base_dir.exists():
            import shutil
            shutil.rmtree(self.base_dir)

class PosixAdapter(BaseStore):
    def __init__(self, name, tag):
        self.base_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_lammps_{tag}")
        if rank == 0:
            if self.base_dir.exists():
                import shutil
                shutil.rmtree(self.base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()
        self.dir = self.base_dir / f"rank_{rank}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bytes_put = 0

    def put(self, key, data):
        p = self.dir / str(key)
        with open(p, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        self.bytes_put += len(data)

    def get(self, key, out):
        p = self.dir / str(key)
        if p.exists():
            with open(p, 'rb') as f:
                out[:] = np.frombuffer(f.read(), dtype=np.uint8)
                return True
        return False

    def storage_used(self):
        return self.bytes_put

    def cleanup(self):
        mpi_barrier()
        if rank == 0 and self.base_dir.exists():
            import shutil
            shutil.rmtree(self.base_dir)

class RedisAdapter(BaseStore):
    def __init__(self, tag):
        import redis
        self.client = redis.Redis(host='localhost', port=16379)
        self.bytes_put = 0

    def put(self, key, data):
        self.client.set(str(key), data.tobytes())
        self.bytes_put += len(data)

    def get(self, key, out):
        val = self.client.get(str(key))
        if val:
            out[:] = np.frombuffer(val, dtype=np.uint8)
            return True
        return False

    def storage_used(self):
        return self.bytes_put

    def cleanup(self):
        try: self.client.flushdb()
        except: pass

def make_adapter(name, tag):
    if name == "Cascade":  return CascadeAdapter(tag)
    elif name == "HDF5":   return HDF5Adapter(tag)
    elif name == "vLLM-GPU": return PosixAdapter("vllm", tag)
    elif name == "PDC":      return PosixAdapter("pdc", tag)
    elif name == "LMCache":  return PosixAdapter("lmcache", tag)
    elif name == "Redis":    return RedisAdapter(tag)
    return None

# ==============================================================================
# LAMMPS I/O Kernel Proxy
# ==============================================================================
# Real LAMMPS writes trajectory dumps containing:
# - Atom ID (int64)
# - Atom Type (int32)
# - Coordinates x, y, z (float64)
# - Velocities vx, vy, vz (float64)
# - Forces fx, fy, fz (float64)
# Total: 8 + 4 + 24 + 24 + 24 = 84 bytes per atom.
# To simulate realistic scale, we target N million atoms per rank.

def run_lammps_io(args):
    systems = args.systems.split(",")
    atoms_per_rank = args.atoms_per_rank
    timesteps = args.timesteps
    
    # Pre-generate LAMMPS data arrays
    # 84 bytes per atom * atoms_per_rank
    bytes_per_atom = 84
    MB_per_rank = (atoms_per_rank * bytes_per_atom) / (1024**2)
    GB_per_rank = MB_per_rank / 1024
    
    print_rank0("\n" + f"{'#'*80}")
    print_rank0("  " + f"REAL WORKLOAD PROXY 1: LAMMPS Molecular Dynamics I/O")
    print_rank0(f"{'#'*80}")
    print_rank0("  " + f"Simulating NVT Ensemble Trajectory Dumps")
    print_rank0("  " + f"Atoms per Rank: {atoms_per_rank:,} (Approx {MB_per_rank:.1f} MB/rank)")
    print_rank0("  " + f"Total Atoms:    {atoms_per_rank * world:,}")
    print_rank0("  " + f"Total Volume:   {GB_per_rank * world * timesteps:.2f} GB (over {timesteps} steps)")
    print_rank0(f"{'#'*80}\n")
    
    # Initialize particle state
    np.random.seed(42 + rank)
    atom_id = np.arange(rank * atoms_per_rank, (rank + 1) * atoms_per_rank, dtype=np.int64)
    atom_type = np.random.randint(1, 4, size=atoms_per_rank, dtype=np.int32)
    pos = np.random.uniform(0, 100, size=(atoms_per_rank, 3)).astype(np.float64)
    vel = np.random.normal(0, 1, size=(atoms_per_rank, 3)).astype(np.float64)
    force = np.random.normal(0, 0.1, size=(atoms_per_rank, 3)).astype(np.float64)

    # To byte arrays
    def state_to_bytes():
        return np.concatenate([
            atom_id.view(np.uint8),
            atom_type.view(np.uint8),
            pos.flatten().view(np.uint8),
            vel.flatten().view(np.uint8),
            force.flatten().view(np.uint8)
        ])

    print_rank0(f"{'System':10} | {'Write BW':>12} | {'Restart BW':>12} | {'Storage Used':>12}")
    print_rank0("-" * 60)

    for sys_name in systems:
        adapter = make_adapter(sys_name, f"lammps_{atoms_per_rank}")
        if not adapter: continue

        total_write_time = 0
        total_write_bytes = 0

        # ---- PHASE 1: Trajectory Dump (Write) ----
        for t in range(timesteps):
            # Simulate MD step: update pos and vel slightly
            pos += vel * 0.001
            vel += force * 0.001
            
            payload = state_to_bytes()
            key_id = f"lammps_dump_step{t}_rank{rank}"
            
            adapter.barrier()
            t0 = time.time()
            if sys_name == "Cascade":
                # LAMMPS relies heavily on unique data per rank (low dedup across ranks for positions)
                # We use raw key for putting.
                import hashlib
                h = hashlib.sha256(payload).hexdigest()
                adapter.put(h, payload)
                # Normally we'd store the hash mapping, for benchmarking we skip it and re-hash on read
            else:
                adapter.put(key_id, payload)
            adapter.barrier()
            
            total_write_time += (time.time() - t0)
            total_write_bytes += len(payload)
            
        write_bw = (total_write_bytes / 1024**3 * world) / total_write_time if total_write_time > 0 else 0

        # ---- PHASE 2: Restart Simulation (Read last step) ----
        if hasattr(adapter, 'sync'): adapter.sync()
        if hasattr(adapter, 'open_for_read'): adapter.open_for_read()
        
        last_step = timesteps - 1
        payload = state_to_bytes() # to get the size and hash
        key_id = f"lammps_dump_step{last_step}_rank{rank}"
        
        buf = np.empty(len(payload), dtype=np.uint8)
        
        adapter.barrier()
        t0 = time.time()
        if sys_name == "Cascade":
            import hashlib
            h = hashlib.sha256(payload).hexdigest()
            adapter.get(h, buf)
        else:
            adapter.get(key_id, buf)
        adapter.barrier()
        
        read_time = time.time() - t0
        read_bw = (len(payload) / 1024**3 * world) / read_time if read_time > 0 else 0
        
        storage_gb = (adapter.storage_used() * world) / 1024**3
        
        print_rank0(f"{sys_name:10} | {write_bw:10.2f} GB/s | {read_bw:10.2f} GB/s | {storage_gb:10.2f} GB")
        
        adapter.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms_per_rank", type=int, default=10000000, help="Number of atoms per rank (10M = ~840MB)")
    parser.add_argument("--timesteps", type=int, default=10, help="Number of dump steps")
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache,Redis", help="Comma-separated storage systems")
    args = parser.parse_args()
    
    _ = init_global_store()
    mpi_barrier()
    
    run_lammps_io(args)
