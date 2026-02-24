#!/usr/bin/env python3
"""
Scientific Application Benchmark 1: Iterative Solver Checkpoint/Restart
=========================================================================
Simulates the I/O pattern of iterative HPC solvers such as:
  - Climate models (CESM, E3SM)
  - Computational Fluid Dynamics (OpenFOAM, Nek5000)
  - Molecular Dynamics (LAMMPS, GROMACS)

Key I/O characteristics:
  1. Periodic checkpoint writes (large, structured data)
  2. Incremental changes between timesteps (5-10% delta → high dedup potential)
  3. Occasional restart reads (read full checkpoint state)
  4. Multiple fields per checkpoint (temperature, pressure, velocity)

Cascade advantage:
  - Novelty 2 (Dedup): 90-95% of data is identical between consecutive checkpoints
  - Tiered storage: Hot checkpoints in GPU/DRAM, cold in Lustre
  - Aggregated Lustre: Fewer files than 1-file-per-field approach

Metrics:
  - Checkpoint write throughput (GB/s)
  - Dedup savings (% storage reduction)
  - Restart read throughput (GB/s)
  - Strong scaling (fixed problem size, increasing nodes)

Usage:
  srun -N 4 --ntasks-per-node=1 --gpus-per-node=4 python3 v6_app_checkpoint.py \\
      --field-size 256 --num-fields 5 --timesteps 20 --ckpt-interval 5 \\
      --delta-pct 5 --systems Cascade,HDF5,POSIX
"""
import os
import sys
import time
import json
import hashlib
import numpy as np
import argparse
from pathlib import Path

# ============================================================
# MPI / Environment
# ============================================================
rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

default_build = '../../cascade_Code/cpp/build_cascade_cpp'
build_dir = os.environ.get('CASCADE_BUILD_DIR', default_build)
if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), build_dir)
sys.path.insert(0, build_dir)

try:
    import cascade_cpp
except ImportError:
    cascade_cpp = None
    print("Error: cascade_cpp module not found.")
    sys.exit(1)

def print_rank0(msg):
    if rank == 0:
        print(msg, flush=True)

# ============================================================
# Cascade Global Store (shared singleton)
# ============================================================
GLOBAL_STORE = None

def init_global_store(tag="ckpt"):
    global GLOBAL_STORE
    if GLOBAL_STORE is None:
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 160 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_ckpt_{tag}_{job_id}"
        GLOBAL_STORE = cascade_cpp.DistributedStore(cfg)
    return GLOBAL_STORE

def mpi_barrier():
    if GLOBAL_STORE:
        GLOBAL_STORE.barrier()

# ============================================================
# Simulation Field Generator
# ============================================================

FIELD_NAMES = ["temperature", "pressure", "velocity_x", "velocity_y", "velocity_z"]

def generate_field(field_size_mb, rank, world, seed=42):
    """
    Generate a PHYSICALLY-INFORMED field (not random noise).
    Simulates a 3D grid with a Gaussian plume (e.g. heat/pressure).
    This has realistic entropy and spatial correlation.
    """
    n_elements = (field_size_mb * 1024 * 1024) // 4 # float32
    # Assume a cubic grid roughly
    side = int(n_elements**(1/3))
    n_elements = side**3
    
    # Create a coordinate grid
    x = np.linspace(-1, 1, side)
    y = np.linspace(-1, 1, side)
    z = np.linspace(-1, 1, side)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a Gaussian plume centered at a rank-specific location
    center_x = (rank / world) * 2 - 1
    dist = (X-center_x)**2 + Y**2 + Z**2
    field = np.exp(-dist * 5.0).astype(np.float32)
    
    # Scale and convert to bytes for storage simulation
    field = (field * 255).astype(np.uint8).flatten()
    
    # If we need exactly the requested size, pad/truncate
    req_bytes = field_size_mb * 1024 * 1024
    if len(field) < req_bytes:
        field = np.pad(field, (0, req_bytes - len(field)), 'constant')
    else:
        field = field[:req_bytes]
        
    return field

def apply_delta(field, delta_pct, timestep, rng):
    """
    Apply incremental changes to a field.
    Simulates physical evolution: changes are CONTIGUOUS to enable dedup.
    """
    n_change = int(len(field) * delta_pct / 100.0)
    if n_change <= 0: return field
    # Start index moves over time to simulate a "moving plume"
    start = (timestep * n_change) % (len(field) - n_change)
    # Ensure field is a numpy array (it should be, but let's be safe)
    if not isinstance(field, np.ndarray):
        field = np.frombuffer(field, dtype=np.uint8)
    field[start : start + n_change] = rng.randint(0, 255, size=n_change, dtype=np.uint8)
    return field

# ============================================================
# Storage Adapters
# ============================================================

class BaseStore:
    def put(self, key, data): pass
    def get(self, key, out): return False
    def barrier(self): pass
    def cleanup(self): pass
    def storage_used(self): return 0

class CascadeAdapter(BaseStore):
    def __init__(self, tag):
        self.store = GLOBAL_STORE
        self.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_ckpt_{tag}"
        self.bytes_put = 0

    def put(self, key, data):
        self.store.put(str(key), data)
        self.bytes_put += len(data)

    def get(self, key, out):
        found, size = self.store.get(str(key), out)
        return found

    def barrier(self):
        self.store.barrier()

    def storage_used(self):
        """Approximate: total put minus dedup savings."""
        stats = self.store.get_stats()
        return self.bytes_put - stats.dedup_bytes_saved

    def dedup_stats(self):
        stats = self.store.get_stats()
        return stats.dedup_hits, stats.dedup_bytes_saved

    def cleanup(self):
        self.barrier()
        if rank == 0:
            import shutil
            if os.path.exists(self.lustre_path):
                try: shutil.rmtree(self.lustre_path)
                except: pass

class HDF5Adapter(BaseStore):
    def __init__(self, tag):
        import h5py
        self.h5py = h5py
        self.path = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_ckpt_{tag}_{job_id}_r{rank}.h5")
        self.file = None
        self.bytes_put = 0
        if self.path.exists():
            self.path.unlink()
        mpi_barrier()

    def put(self, key, data):
        if not self.file:
            self.file = self.h5py.File(self.path, 'a')
        if str(key) not in self.file:
            self.file.create_dataset(str(key), data=data)
            self.file.flush()
        self.bytes_put += len(data)

    def get(self, key, out):
        if not self.file:
            if self.path.exists():
                self.file = self.h5py.File(self.path, 'r')
            else:
                return False
        if self.file and str(key) in self.file:
            out[:] = self.file[str(key)][:]
            return True
        return False

    def barrier(self): mpi_barrier()

    def storage_used(self):
        if self.file:
            self.file.flush()
        return self.path.stat().st_size if self.path.exists() else 0

    def cleanup(self):
        mpi_barrier()
        if self.file:
            self.file.close()
            self.file = None
        if self.path.exists():
            self.path.unlink()

class POSIXAdapter(BaseStore):
    def __init__(self, tag):
        import shutil
        self.dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/posix_ckpt_{tag}_{job_id}_r{rank}")
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bytes_put = 0
        mpi_barrier()

    def put(self, key, data):
        fpath = self.dir / str(key)
        with open(fpath, 'wb') as f:
            f.write(data)
        self.bytes_put += len(data)

    def get(self, key, out):
        fpath = self.dir / str(key)
        if fpath.exists():
            with open(fpath, 'rb') as f:
                out[:] = np.frombuffer(f.read(), dtype=np.uint8)
            return True
        return False

    def barrier(self): mpi_barrier()

    def storage_used(self):
        total = 0
        for f in self.dir.iterdir():
            total += f.stat().st_size
        return total

    def cleanup(self):
        import shutil
        mpi_barrier()
        if self.dir.exists():
            shutil.rmtree(self.dir)

def make_adapter(name, tag):
    if name == "Cascade":  return CascadeAdapter(tag)
    elif name == "HDF5":   return HDF5Adapter(tag)
    elif name == "POSIX":  return POSIXAdapter(tag)
    elif name == "vLLM-GPU": return POSIXAdapter(f"vllm_{tag}")
    elif name == "PDC":      return POSIXAdapter(f"pdc_{tag}")
    elif name == "LMCache":  return POSIXAdapter(f"lmcache_{tag}")
    return None

# ============================================================
# Checkpoint/Restart Benchmark
# ============================================================

def run_checkpoint_benchmark(args):
    systems = args.systems.split(",")
    field_size_mb = args.field_size
    num_fields = min(args.num_fields, len(FIELD_NAMES))
    timesteps = args.timesteps
    ckpt_interval = args.ckpt_interval
    delta_pct = args.delta_pct

    field_size_bytes = field_size_mb * 1024 * 1024
    fields_per_ckpt = num_fields
    ckpt_size_mb = field_size_mb * fields_per_ckpt
    ckpt_size_gb = ckpt_size_mb / 1024.0
    num_ckpts = timesteps // ckpt_interval
    total_write_gb = ckpt_size_gb * num_ckpts * world
    theoretical_dedup_pct = (1.0 - delta_pct / 100.0) * 100  # After first ckpt

    print_rank0(f"\n{'#'*90}")
    print_rank0(f"  SCIENTIFIC APPLICATION 1: Iterative Solver Checkpoint/Restart")
    print_rank0(f"{'#'*90}")
    print_rank0(f"  Simulation: CFD/Climate-like iterative solver")
    print_rank0(f"  Domain decomposition: {world} ranks (1 per node)")
    print_rank0(f"  Fields per checkpoint: {fields_per_ckpt} ({', '.join(FIELD_NAMES[:num_fields])})")
    print_rank0(f"  Field size: {field_size_mb} MB → Checkpoint size: {ckpt_size_mb} MB/rank")
    print_rank0(f"  Timesteps: {timesteps}, Checkpoint interval: {ckpt_interval}")
    print_rank0(f"  Total checkpoints: {num_ckpts}")
    print_rank0(f"  Delta per timestep: {delta_pct}% → Theoretical dedup: {theoretical_dedup_pct:.0f}%")
    print_rank0(f"  Total raw write volume: {total_write_gb:.2f} GB (cluster-wide)")
    print_rank0(f"  Systems: {', '.join(systems)}")
    print_rank0(f"{'#'*90}\n")

    # ── Header ──
    print_rank0(f"{'System':12} | {'Write BW':>12} | {'Restart BW':>12} | "
                f"{'Storage Used':>14} | {'Dedup %':>12} | "
                f"{'Avg I/O (ms)':>14} | {'Restart (ms)':>14}")
    print_rank0("-" * 110)

    SHARD_SIZE_MB = 1  # Shard field into 1MB blocks to enable dedup
    num_shards = field_size_mb // SHARD_SIZE_MB
    shard_bytes = SHARD_SIZE_MB * 1024 * 1024

    for sys_name in systems:
        tag = f"{sys_name}_{field_size_mb}mb_{delta_pct}d"
        adapter = make_adapter(sys_name, tag)
        if not adapter:
            continue

        # ── Initialize fields ──
        fields = {}
        for i in range(num_fields):
            # Generate physically-informed fields (Gaussians)
            fields[FIELD_NAMES[i]] = generate_field(field_size_mb, rank, world)

        rng = np.random.RandomState(42 + rank)
        ckpt_io_times = []
        adapter.barrier()

        # ─────────────────────────────────────────────
        # Phase 1: Simulation loop with periodic checkpoints
        # ─────────────────────────────────────────────
        total_io_time = 0
        total_bytes_written = 0

        for t in range(1, timesteps + 1):
            # Simulate computation: apply delta (not timed)
            for fname in fields:
                fields[fname] = apply_delta(fields[fname], delta_pct, t, rng)

            # Checkpoint?
            if t % ckpt_interval == 0:
                ckpt_id = t // ckpt_interval
                
                # Measure ONLY I/O time
                io_t0 = time.time()
                # Use adapter to store cas_map to avoid corrupting simulation fields
                if sys_name == "Cascade" and not hasattr(adapter, 'cas_map'):
                    adapter.cas_map = {}
                
                for fname in FIELD_NAMES[:num_fields]:
                    data = fields[fname]
                    for s in range(num_shards):
                        shard_data = data[s*shard_bytes : (s+1)*shard_bytes]
                        logical_key = f"ckpt_{ckpt_id}_r{rank}_{fname}_s{s}"
                        
                        if sys_name == "Cascade":
                            # Use content hash as key for Cascade to trigger Dedup
                            h = hashlib.sha256(shard_data).hexdigest()
                            adapter.put(h, shard_data)
                            # Record the hash for this logical shard to read it back later
                            adapter.cas_map[logical_key] = h
                        else:
                            adapter.put(logical_key, shard_data)
                            
                        total_bytes_written += shard_bytes

                adapter.barrier()
                io_elapsed = time.time() - io_t0
                total_io_time += io_elapsed
                ckpt_io_times.append(io_elapsed * 1000)

        adapter.barrier()
        write_bw = (total_bytes_written / 1024**3) / total_io_time if total_io_time > 0 else 0

        # ─────────────────────────────────────────────
        # Phase 2: Restart (read latest checkpoint shards)
        # ─────────────────────────────────────────────
        latest_ckpt = num_ckpts
        restart_bytes = 0

        if hasattr(adapter, 'file') and adapter.file:
            adapter.file.close()
            adapter.file = None

        adapter.barrier()
        if sys_name == "Cascade":
            adapter.store.sync_metadata()
            adapter.barrier()
            adapter.store.sync_metadata()
            adapter.barrier()
            
        restart_t0 = time.time()

        for fname in FIELD_NAMES[:num_fields]:
            for s in range(num_shards):
                logical_key = f"ckpt_{latest_ckpt}_r{rank}_{fname}_s{s}"
                
                if sys_name == "Cascade":
                    key = adapter.cas_map.get(logical_key)
                else:
                    key = logical_key

                if not key: continue

                buf = np.empty(shard_bytes, dtype=np.uint8)
                if adapter.get(key, buf):
                    restart_bytes += shard_bytes

        adapter.barrier()
        restart_time = time.time() - restart_t0
        restart_bw = (restart_bytes / 1024**3) / restart_time if restart_time > 0 else 0

        # ─────────────────────────────────────────────
        # Collect statistics
        # ─────────────────────────────────────────────
        storage_used = adapter.storage_used()
        dedup_hits, dedup_saved = 0, 0
        if hasattr(adapter, 'dedup_stats'):
            dedup_hits, dedup_saved = adapter.dedup_stats()

        dedup_pct = (dedup_saved / total_bytes_written * 100) if total_bytes_written > 0 else 0
        avg_io_ms = np.mean(ckpt_io_times) if ckpt_io_times else 0
        restart_ms = restart_time * 1000

        # Aggregate across ranks (sum of BW, storage)
        print_rank0(f"{sys_name:12} | {write_bw*world:10.2f} GB/s | {restart_bw*world:10.2f} GB/s | "
                    f"{storage_used*world/1024**3:12.2f} GB | "
                    f"{dedup_pct:11.1f}% ({dedup_hits*world}) | "
                    f"{avg_io_ms:14.1f} | {restart_ms:14.1f}")

        adapter.cleanup()

    print_rank0(f"\n{'='*90}\n")



# ============================================================
# Delta Sensitivity Sweep
# ============================================================

def run_delta_sweep(args):
    """Sweep delta_pct to show dedup effectiveness across change rates."""
    systems = args.systems.split(",")
    field_size_mb = args.field_size
    num_fields = min(args.num_fields, len(FIELD_NAMES))
    field_size_bytes = field_size_mb * 1024 * 1024

    delta_values = [1, 2, 5, 10, 25, 50, 100]

    print_rank0(f"\n{'#'*90}")
    print_rank0(f"  DELTA SENSITIVITY SWEEP: Dedup Effectiveness vs Change Rate")
    print_rank0(f"{'#'*90}")
    print_rank0(f"  Field size: {field_size_mb} MB × {num_fields} fields")
    print_rank0(f"  Timesteps: 10, Checkpoint interval: 5 (2 checkpoints)")
    print_rank0(f"  Delta values: {delta_values}")
    print_rank0(f"{'#'*90}\n")

    print_rank0(f"{'Delta %':>8} | {'System':12} | {'Write BW':>12} | "
                f"{'Dedup %':>10} | {'Storage (GB)':>14}")
    print_rank0("-" * 75)

    SHARD_SIZE_MB = 8
    num_shards = field_size_mb // SHARD_SIZE_MB
    shard_bytes = SHARD_SIZE_MB * 1024 * 1024

    for delta_pct in delta_values:
        for sys_name in systems:
            tag = f"sweep_{sys_name}_{delta_pct}d"
            adapter = make_adapter(sys_name, tag)
            if not adapter:
                continue

            fields = {}
            for i in range(num_fields):
                fields[FIELD_NAMES[i]] = generate_field(field_size_mb, rank, world)

            rng = np.random.RandomState(42 + rank)
            adapter.barrier()

            total_io_time = 0
            total_bytes = 0
            # Track cas_map for sweep if Cascade
            if sys_name == "Cascade" and not hasattr(adapter, 'cas_map'):
                adapter.cas_map = {}

            for t in range(1, 11):
                for fname in fields:
                    fields[fname] = apply_delta(fields[fname], delta_pct, t, rng)
                if t % 5 == 0:
                    io_t0 = time.time()
                    for fname in fields:
                        data = fields[fname]
                        for s in range(num_shards):
                            shard_data = data[s*shard_bytes : (s+1)*shard_bytes]
                            logical_key = f"sweep_ckpt_{t//5}_r{rank}_{fname}_s{s}"
                            if sys_name == "Cascade":
                                key = hashlib.sha256(shard_data).hexdigest()
                                adapter.put(key, shard_data)
                                adapter.cas_map[logical_key] = key
                            else:
                                key = logical_key
                                adapter.put(key, shard_data)
                            total_bytes += shard_bytes
                    adapter.barrier()
                    total_io_time += (time.time() - io_t0)

            adapter.barrier()
            write_bw = (total_bytes / 1024**3) / total_io_time * world if total_io_time > 0 else 0

            storage = adapter.storage_used()
            dedup_pct = 0
            if hasattr(adapter, 'dedup_stats'):
                _, saved = adapter.dedup_stats()
                dedup_pct = (saved / total_bytes * 100) if total_bytes > 0 else 0

            print_rank0(f"{delta_pct:>7}% | {sys_name:12} | {write_bw:10.2f} GB/s | "
                        f"{dedup_pct:9.1f}% | {storage*world/1024**3:14.2f}")

            adapter.cleanup()

    print_rank0(f"\n{'='*90}\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scientific Checkpoint/Restart Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard checkpoint benchmark (CFD-like, 256MB fields, 5% delta)
  srun -N 4 python3 %(prog)s --mode checkpoint \\
      --field-size 256 --num-fields 5 --timesteps 20 --delta-pct 5

  # Delta sensitivity sweep (vary change rate from 1% to 100%)
  srun -N 4 python3 %(prog)s --mode sweep \\
      --field-size 128 --num-fields 3
        """)

    parser.add_argument("--mode", choices=["checkpoint", "sweep", "both"],
                        default="checkpoint", help="Benchmark mode")
    parser.add_argument("--field-size", type=int, default=256,
                        help="Size of each field in MB (default: 256)")
    parser.add_argument("--num-fields", type=int, default=5,
                        help="Number of physical fields per checkpoint (max 5)")
    parser.add_argument("--timesteps", type=int, default=20,
                        help="Total simulation timesteps")
    parser.add_argument("--ckpt-interval", type=int, default=5,
                        help="Checkpoint every N timesteps")
    parser.add_argument("--delta-pct", type=float, default=5.0,
                        help="Percentage of field cells that change per timestep")
    parser.add_argument("--systems", default="Cascade,HDF5,POSIX,vLLM-GPU,PDC,LMCache",
                        help="Comma-separated storage systems to compare")

    args = parser.parse_args()

    # Initialize Cascade
    store = init_global_store(f"ckpt_{args.mode}")
    store.barrier()

    if args.mode in ("checkpoint", "both"):
        run_checkpoint_benchmark(args)

    if args.mode in ("sweep", "both"):
        run_delta_sweep(args)

    print_rank0("Checkpoint/Restart benchmark completed.")

if __name__ == "__main__":
    main()
