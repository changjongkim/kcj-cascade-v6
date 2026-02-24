import os
import sys
import time
import argparse
import numpy as np
import hashlib

rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))

# Path to built cascade_cpp.so
sys.path.append("/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build")

try:
    import cascade_cpp
except ImportError:
    if rank == 0:
        print("Error: Could not import cascade_cpp.")
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

    def put(self, key, data, is_pf=False):
        self.store.put(key, data, is_pf)
        self.bytes_put += len(data)

    def get(self, key, out):
        return self.store.get(key, out)

    def sync(self):
        self.store.sync_metadata()
        self.barrier()
        self.store.sync_metadata()
        self.barrier()

    def storage_used(self):
        stats = self.store.get_stats()
        return self.bytes_put - stats.dedup_bytes_saved

import h5py
from pathlib import Path

class HDF5Adapter(BaseStore):
    def __init__(self, tag, file_per_rank=True):
        self.base_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_amrex_{tag}")
        if rank == 0:
            if self.base_dir.exists():
                import shutil
                try: shutil.rmtree(self.base_dir)
                except: pass
            self.base_dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()
        time.sleep(1)
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
        mpi_barrier()
        time.sleep(1)
        self.file = h5py.File(self.filepath, 'r')
        self.read_mode = True

    def get(self, key, out):
        if not self.read_mode:
            self.open_for_read()
            
        if isinstance(key, str) and "_rank" in key:
            rk = int(key.split("_rank")[1].split("_")[0])
            if rk != rank:
                neighbor_file = self.base_dir / f"rank_{rk}.h5"
                if neighbor_file.exists():
                    try:
                        with h5py.File(neighbor_file, 'r') as f:
                            if key in f:
                                out[:] = f[key][:]
                                return True
                    except Exception:
                        return False
                return False

        if key in self.file:
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
        self.base_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/{name}_amrex_{tag}")
        if rank == 0:
            if self.base_dir.exists():
                import shutil
                try: shutil.rmtree(self.base_dir)
                except: pass
            self.base_dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()
        time.sleep(1)
        self.dir = self.base_dir / f"rank_{rank}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bytes_put = 0

    def put(self, key, data, is_pf=False):
        p = self.dir / str(key)
        with open(p, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        self.bytes_put += len(data)

    def get(self, key, out):
        p = self.dir / str(key)
        if isinstance(key, str) and "_rank" in key:
            rk = int(key.split("_rank")[1].split("_")[0])
            if rk != rank:
                p = self.base_dir / f"rank_{rk}" / str(key)
                
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

    def put(self, key, data, is_pf=False):
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
# AMReX I/O Kernel Proxy
# ==============================================================================
# AMReX structured grids with hierarchical refinement.
# Generates "Plotfiles" containing MultiFabs.
# Variables typically tracked: density, xmom, ymom, zmom, energy
# For a grid of [nx, ny, nz] per block, each scalar is float64 (8 bytes).
# To mimic AMReX boundary exchange, nodes request overlapping ghost cells (halo exchange).

def run_amrex_io(args):
    systems = args.systems.split(",")
    n_blocks = args.n_blocks # Blocks per rank
    grid_dim = args.grid_dim # (N, N, N) grid size per block
    timesteps = args.timesteps
    
    # Grid variables (5 physical quantities)
    vars = ['rho', 'mx', 'my', 'mz', 'E']
    
    bytes_per_block = grid_dim**3 * len(vars) * 8 # float64
    MB_per_rank = (n_blocks * bytes_per_block) / (1024**2)
    GB_per_rank = MB_per_rank / 1024
    
    print_rank0("\n" + f"{'#'*80}")
    print_rank0("  " + f"REAL WORKLOAD PROXY 2: AMReX Adaptive Mesh Refinement I/O")
    print_rank0(f"{'#'*80}")
    print_rank0("  " + f"Simulating AMReX MultiFab Plotfile Dumps & Halo Exchange")
    print_rank0("  " + f"Grid Size per Block: {grid_dim}x{grid_dim}x{grid_dim} (5 Variables)")
    print_rank0("  " + f"Blocks per Rank: {n_blocks} (Approx {MB_per_rank:.1f} MB/rank)")
    print_rank0("  " + f"Total Volume:    {GB_per_rank * world * timesteps:.2f} GB (over {timesteps} steps)")
    print_rank0(f"{'#'*80}\n")
    
    # Generate static multifabs with some physics-like noise
    multifabs = {}
    np.random.seed(42 + rank)
    for b in range(n_blocks):
        grid = {}
        for var in vars:
            grid[var] = np.random.normal(loc=1.0, scale=0.1, size=(grid_dim, grid_dim, grid_dim)).astype(np.float64)
        multifabs[b] = grid

    def get_block_bytes(b):
        parts = [multifabs[b][v].tobytes() for v in vars]
        return np.concatenate([np.frombuffer(p, dtype=np.uint8) for p in parts])

    print_rank0(f"{'System':10} | {'Plotfile BW':>12} | {'Restart BW':>12} | {'Halo Ex BW':>12}")
    print_rank0("-" * 65)

    for sys_name in systems:
        adapter = make_adapter(sys_name, f"amrex_g{grid_dim}_b{n_blocks}")
        if not adapter: continue

        total_write_time = 0
        total_write_bytes = 0

        # ---- PHASE 1: Plotfile Dumps ----
        for t in range(timesteps):
            for b in range(n_blocks):
                # Apply tiny delta to mimic physics evolution
                for var in vars:
                    multifabs[b][var] += 0.001
                
                payload = get_block_bytes(b)
                key_id = f"amrex_plotfile_step{t}_rank{rank}_blk{b}"
                
                adapter.barrier()
                t0 = time.time()
                if sys_name == "Cascade":
                    h = hashlib.sha256(payload).hexdigest()
                    adapter.put(h, payload)
                else:
                    adapter.put(key_id, payload)
                adapter.barrier()
                
                total_write_time += (time.time() - t0)
                total_write_bytes += len(payload)
                
        plot_bw = (total_write_bytes / 1024**3 * world) / total_write_time if total_write_time > 0 else 0

        # ---- PHASE 2: Restart from last Plotfile ----
        if hasattr(adapter, 'sync'): adapter.sync()
        if hasattr(adapter, 'open_for_read'): adapter.open_for_read()
        
        last_step = timesteps - 1
        restart_time = 0
        buf = np.empty(bytes_per_block, dtype=np.uint8)
        
        for b in range(n_blocks):
            payload = get_block_bytes(b)
            key_id = f"amrex_plotfile_step{last_step}_rank{rank}_blk{b}"
            
            adapter.barrier()
            t0 = time.time()
            if sys_name == "Cascade":
                h = hashlib.sha256(payload).hexdigest()
                adapter.get(h, buf)
            else:
                adapter.get(key_id, buf)
            adapter.barrier()
            restart_time += (time.time() - t0)

        restart_bw = (n_blocks * bytes_per_block / 1024**3 * world) / restart_time if restart_time > 0 else 0

        # ---- PHASE 3: Halo (Ghost Cell) Exchange (Reading Neighbours Blocks) ----
        # Rank reads block 0 from Right Rank and Left Rank
        halo_time = 0
        halo_bytes = 0
        
        left = (rank - 1) % world
        right = (rank + 1) % world
        
        for neighbor in [left, right]:
            if neighbor == rank: continue
            
            # Using same pseudo payload to get hash
            np.random.seed(42 + neighbor)
            ghost_payload = bytes_per_block
            key_id = f"amrex_plotfile_step{last_step}_rank{neighbor}_blk0"
            
            # To fetch from cascade we need the hash. In real system metadata maintains this map.
            # We just regenerate the theoretical payload locally to find hash ID.
            dummy_grid = {}
            for var in vars:
                dummy_grid[var] = np.random.normal(loc=1.0, scale=0.1, size=(grid_dim, grid_dim, grid_dim)).astype(np.float64)
                # Apply timesteps of physics evolution to get exact hash
                for _ in range(timesteps):
                    dummy_grid[var] += 0.001
            parts = [dummy_grid[var].tobytes() for var in vars]
            expected_payload = np.concatenate([np.frombuffer(p, dtype=np.uint8) for p in parts])
            h = hashlib.sha256(expected_payload).hexdigest()
            
            adapter.barrier()
            t0 = time.time()
            if sys_name == "Cascade":
                success = adapter.get(h, buf)
            else:
                success = adapter.get(key_id, buf)
            adapter.barrier()
            
            if success:
                halo_bytes += bytes_per_block
            halo_time += (time.time() - t0)
            
        halo_bw = (halo_bytes / 1024**3 * world) / halo_time if halo_time > 0 else 0
        
        print_rank0(f"{sys_name:10} | {plot_bw:10.2f} GB/s | {restart_bw:10.2f} GB/s | {halo_bw:10.2f} GB/s")
        adapter.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_dim", type=int, default=128, help="Grid size per block (e.g. 128x128x128)")
    parser.add_argument("--n_blocks", type=int, default=10, help="Number of multi-level blocks per rank")
    parser.add_argument("--timesteps", type=int, default=10, help="Number of dump steps")
    parser.add_argument("--systems", default="Cascade,HDF5,vLLM-GPU,PDC,LMCache", help="Comma-separated storage systems")
    args = parser.parse_args()
    
    _ = init_global_store()
    mpi_barrier()
    
    run_amrex_io(args)
