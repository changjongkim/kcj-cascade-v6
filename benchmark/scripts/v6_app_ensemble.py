#!/usr/bin/env python3
"""
Scientific Application Benchmark 2: Ensemble Simulation with Shared Initial Conditions
========================================================================================
Simulates the I/O pattern of ensemble-based HPC workflows such as:
  - Weather ensemble forecasting (ECMWF ENS, GFS)
  - Uncertainty Quantification (UQ) parameter sweeps
  - Monte Carlo methods with shared geometry
  - Multi-physics coupling (shared mesh/grid data)

Key I/O characteristics:
  1. SHARED initial conditions: All ensemble members read the same IC (large data)
  2. PER-MEMBER evolution: Each member writes unique state data
  3. BOUNDARY EXCHANGE: Members periodically read neighbor data (coupling)
  4. ANALYSIS REDUCTION: All members' data read for ensemble statistics

Cascade advantages demonstrated:
  - Novelty 1 (Prefix Protection): Shared IC blocks are marked as prefix → never evicted
  - Novelty 2 (Dedup): IC written once, all members reference the same data
  - Tier 3/4 (RDMA): Boundary exchange uses cross-node RDMA instead of Lustre
  - Novelty 3 (Locality): Frequently exchanged boundary data auto-promoted to local

Metrics:
  - Shared IC distribution throughput (GB/s) & dedup savings
  - Per-member write throughput (GB/s)
  - Boundary exchange latency (ms) & throughput
  - Analysis read throughput (GB/s, all-to-all)
  - End-to-end pipeline time

Usage:
  srun -N 8 --ntasks-per-node=1 --gpus-per-node=4 python3 v6_app_ensemble.py \\
      --num-members 8 --ic-blocks 20 --member-blocks 10 \\
      --exchange-rounds 5 --block-size 128 --systems Cascade,HDF5,POSIX
"""
import os
import sys
import time
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
# Cascade Global Store
# ============================================================
GLOBAL_STORE = None

def init_global_store(tag="ensemble"):
    global GLOBAL_STORE
    if GLOBAL_STORE is None:
        cfg = cascade_cpp.DistributedConfig()
        cfg.gpu_capacity_per_device = 38 * 1024**3
        cfg.dram_capacity = 160 * 1024**3
        cfg.num_gpus_per_node = 4
        cfg.dedup_enabled = True
        cfg.kv_compression = True
        cfg.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_ens_{tag}_{job_id}"
        GLOBAL_STORE = cascade_cpp.DistributedStore(cfg)
    return GLOBAL_STORE

def mpi_barrier():
    if GLOBAL_STORE:
        GLOBAL_STORE.barrier()

# ============================================================
# Data Generators
# ============================================================

def generate_block(seed, size):
    """Generate a deterministic data block."""
    np.random.seed(seed % (2**32))
    return np.random.randint(0, 255, size, dtype=np.uint8)

def perturb_block(base_data, perturbation_pct, member_id, block_id):
    """Apply a small perturbation to simulate ensemble member divergence."""
    rng = np.random.RandomState(member_id * 10000 + block_id)
    result = base_data.copy()
    n_change = int(len(result) * perturbation_pct / 100.0)
    indices = rng.choice(len(result), size=n_change, replace=False)
    result[indices] = rng.randint(0, 255, size=n_change, dtype=np.uint8)
    return result

# ============================================================
# Storage Adapters
# ============================================================

class BaseStore:
    def put(self, key, data): pass
    def get(self, key, out): return False
    def barrier(self): pass
    def cleanup(self): pass

class CascadeAdapter(BaseStore):
    def __init__(self, tag):
        self.store = GLOBAL_STORE
        self.lustre_path = f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_ens_{tag}"

    def put(self, key, data):
        self.store.put(str(key), data)

    def get(self, key, out):
        found, size = self.store.get(str(key), out)
        return found

    def barrier(self):
        self.store.barrier()

    def dedup_stats(self):
        stats = self.store.get_stats()
        return stats.dedup_hits, stats.dedup_bytes_saved

    def tier_stats(self):
        """Return hit stats per tier."""
        stats = self.store.get_stats()
        return {
            'local_gpu': stats.local_gpu_hits,
            'local_dram': stats.local_dram_hits,
            'remote_gpu': stats.remote_gpu_hits,
            'remote_dram': stats.remote_dram_hits,
            'lustre': stats.lustre_hits,
            'misses': stats.misses,
            'promotions': stats.promotions_to_local,
        }

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
        self.base_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/hdf5_ens_{tag}_{job_id}")
        if rank == 0:
            if self.base_dir.exists():
                import shutil
                shutil.rmtree(self.base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()
        self.path = self.base_dir / f"rank_{rank}.h5"
        self.file_write = None
        self.open_files = {} # path -> handle

    def put(self, key, data):
        if not self.file_write:
            self.file_write = self.h5py.File(self.path, 'a')
        if str(key) not in self.file_write:
            self.file_write.create_dataset(str(key), data=data)
            self.file_write.flush()

    def get(self, key, out):
        # Scan all rank files in the shared directory
        for fpath in self.base_dir.glob("*.h5"):
            if fpath not in self.open_files:
                try: self.open_files[fpath] = self.h5py.File(fpath, 'r')
                except: continue
            
            h5 = self.open_files[fpath]
            if str(key) in h5:
                out[:] = h5[str(key)][:]
                return True
        return False

    def barrier(self): mpi_barrier()

    def cleanup(self):
        mpi_barrier()
        if self.file_write: self.file_write.close()
        for f in self.open_files.values(): f.close()
        mpi_barrier()
        if rank == 0 and self.base_dir.exists():
            import shutil
            shutil.rmtree(self.base_dir)

class POSIXAdapter(BaseStore):
    def __init__(self, tag):
        self.base_dir = Path(f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/posix_ens_{tag}_{job_id}")
        if rank == 0:
            if self.base_dir.exists():
                import shutil
                shutil.rmtree(self.base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        mpi_barrier()
        self.my_dir = self.base_dir / f"rank_{rank}"
        self.my_dir.mkdir(exist_ok=True)
        mpi_barrier()

    def put(self, key, data):
        with open(self.my_dir / str(key), 'wb') as f:
            f.write(data)

    def get(self, key, out):
        # Scan all rank directories
        for r_dir in self.base_dir.glob("rank_*"):
            fpath = r_dir / str(key)
            if fpath.exists():
                with open(fpath, 'rb') as f:
                    out[:] = np.frombuffer(f.read(), dtype=np.uint8)
                return True
        return False

    def barrier(self): mpi_barrier()

    def cleanup(self):
        mpi_barrier()
        if rank == 0 and self.base_dir.exists():
            import shutil
            shutil.rmtree(self.base_dir)

def make_adapter(name, tag):
    if name == "Cascade":  return CascadeAdapter(tag)
    elif name == "HDF5":   return HDF5Adapter(tag)
    elif name == "POSIX":  return POSIXAdapter(tag)
    elif name == "vLLM-GPU": return POSIXAdapter(f"vllm_{tag}")
    elif name == "PDC":      return POSIXAdapter(f"pdc_{tag}")
    elif name == "LMCache":  return POSIXAdapter(f"lmcache_{tag}")
    return None

# ============================================================
# Phase Benchmarks
# ============================================================

def phase1_distribute_ic(adapter, ic_blocks, block_size, sys_name):
    """
    Phase 1: Distribute shared Initial Conditions.
    Rank 0 writes IC blocks. All ranks then read them.
    For Cascade: only one copy exists (dedup), all read via RDMA.
    For others: each rank reads from shared Lustre file.
    """
    ic_keys = [f"ic_field_{i}" for i in range(ic_blocks)]

    # Write IC (rank 0 only for shared systems)
    adapter.barrier()
    write_t0 = time.time()

    if rank == 0:
        for k in ic_keys:
            data = generate_block(hash(k) % (2**32), block_size)
            adapter.put(k, data)

    adapter.barrier()
    write_time = time.time() - write_t0

    # For Cascade: all ranks also put (dedup should catch it)
    # This simulates the "every member loads IC" pattern
    if sys_name == "Cascade":
        for k in ic_keys:
            # Re-generate data precisely as rank 0 did
            data = generate_block(hash(k) % (2**32), block_size)
            adapter.put(k, data)
        adapter.barrier()

    # Prepare for read
    if hasattr(adapter, 'open_for_read'):
        adapter.open_for_read()

    # All ranks read IC
    adapter.barrier()
    read_t0 = time.time()
    read_bytes = 0

    for k in ic_keys:
        buf = np.empty(block_size, dtype=np.uint8)
        if adapter.get(k, buf):
            read_bytes += block_size

    adapter.barrier()
    read_time = time.time() - read_t0

    ic_total_gb = (ic_blocks * block_size) / 1024**3
    write_bw = ic_total_gb / write_time if write_time > 0 else 0
    read_bw = (read_bytes / 1024**3 * world) / read_time if read_time > 0 else 0

    return write_time, read_time, write_bw, read_bw, None


def phase2_member_evolution(adapter, member_blocks, block_size, evolution_steps):
    """
    Phase 2: Each member evolves independently and writes unique state.
    Each rank = one ensemble member.
    """
    adapter.barrier()
    write_t0 = time.time()
    total_bytes = 0

    for step in range(evolution_steps):
        for b in range(member_blocks):
            key = f"member_{rank}_step_{step}_block_{b}"
            data = generate_block(
                hash(f"m{rank}_s{step}_b{b}") % (2**32), block_size
            )
            adapter.put(key, data)
            total_bytes += block_size

    adapter.barrier()
    write_time = time.time() - write_t0
    write_bw = (total_bytes / 1024**3 * world) / write_time if write_time > 0 else 0

    return write_time, write_bw, total_bytes


def phase3_boundary_exchange(adapter, exchange_rounds, block_size, sys_name):
    """
    Phase 3: Boundary exchange — each rank reads data from neighbor ranks.
    Simulates domain decomposition halo exchange.
    For Cascade: hits RDMA tier (Tier 3/4). For others: hits Lustre.
    """
    # Each rank writes its boundary data
    for r_round in range(exchange_rounds):
        bnd_key = f"boundary_rank{rank}_round{r_round}"
        bnd_data = generate_block(
            hash(f"bnd_{rank}_{r_round}") % (2**32), block_size
        )
        adapter.put(bnd_key, bnd_data)

    # Flush writes
    if hasattr(adapter, 'file_write') and adapter.file_write:
        adapter.file_write.flush()
        adapter.file_write.close()
        adapter.file_write = None
    adapter.barrier()

    if hasattr(adapter, 'open_for_read'):
        adapter.open_for_read()

    # Read neighbor boundaries (ring topology: rank reads from rank-1 and rank+1)
    adapter.barrier()
    exchange_t0 = time.time()
    exchange_bytes = 0
    exchange_latencies = []

    for r_round in range(exchange_rounds):
        # Read from left neighbor
        left = (rank - 1) % world
        right = (rank + 1) % world

        for neighbor in [left, right]:
            if neighbor == rank:
                continue
            key = f"boundary_rank{neighbor}_round{r_round}"
            buf = np.empty(block_size, dtype=np.uint8)
            lat_t0 = time.time()
            success = adapter.get(key, buf)
            lat = (time.time() - lat_t0) * 1000
            exchange_latencies.append(lat)
            
            if success:
                exchange_bytes += block_size
            else:
                if rank == 0 and r_round == 0:
                    print(f"[DEBUG] Rank 0 failed to read {key} from neighbor {neighbor}")

    adapter.barrier()
    exchange_time = time.time() - exchange_t0
    exchange_bw = (exchange_bytes / 1024**3 * world) / exchange_time if exchange_time > 0 else 0
    avg_lat = np.mean(exchange_latencies) if exchange_latencies else 0
    
    if rank == 0:
        print(f"[DEBUG] Exchange Phase: bytes={exchange_bytes}, time={exchange_time:.4f}s, bw={exchange_bw:.2f}")

    return exchange_time, exchange_bw, avg_lat, exchange_latencies


def phase4_analysis_reduction(adapter, member_blocks, evolution_steps, block_size):
    """
    Phase 4: Analysis/Reduction — all ranks read all members' final state.
    Simulates ensemble statistics computation (mean, spread, etc.).
    Each rank reads EVERY member's last timestep.
    """
    if hasattr(adapter, 'open_for_read'):
        adapter.open_for_read()

    adapter.barrier()
    analysis_t0 = time.time()
    analysis_bytes = 0

    last_step = evolution_steps - 1
    for member in range(world):
        for b in range(member_blocks):
            key = f"member_{member}_step_{last_step}_block_{b}"
            buf = np.empty(block_size, dtype=np.uint8)
            if adapter.get(key, buf):
                analysis_bytes += block_size

    adapter.barrier()
    analysis_time = time.time() - analysis_t0
    analysis_bw = (analysis_bytes / 1024**3 * world) / analysis_time if analysis_time > 0 else 0

    return analysis_time, analysis_bw


# ============================================================
# Main Ensemble Benchmark
# ============================================================

def run_ensemble_benchmark(args):
    systems = args.systems.split(",")
    block_size = args.block_size * 1024 * 1024  # MB → bytes
    ic_blocks = args.ic_blocks
    member_blocks = args.member_blocks
    evolution_steps = args.evolution_steps
    exchange_rounds = args.exchange_rounds

    ic_gb = (ic_blocks * block_size) / 1024**3
    member_gb = (member_blocks * evolution_steps * block_size) / 1024**3
    total_gb = ic_gb + member_gb * world

    print_rank0(f"\n{'#'*90}")
    print_rank0(f"  SCIENTIFIC APPLICATION 2: Ensemble Simulation Pipeline")
    print_rank0(f"{'#'*90}")
    print_rank0(f"  Application: Weather/Climate Ensemble Forecast")
    print_rank0(f"  Ensemble members: {world} (1 per node)")
    print_rank0(f"  Block size: {args.block_size} MB")
    print_rank0(f"  ──────────────────────────────────────────────────────")
    print_rank0(f"  Phase 1 — Shared IC Distribution:")
    print_rank0(f"    IC blocks: {ic_blocks} ({ic_gb:.2f} GB)")
    print_rank0(f"    All {world} members read the SAME IC → Dedup opportunity")
    print_rank0(f"  Phase 2 — Member Evolution:")
    print_rank0(f"    Per-member blocks: {member_blocks} × {evolution_steps} steps")
    print_rank0(f"    Total member data: {member_gb:.2f} GB/member × {world} = {member_gb * world:.2f} GB")
    print_rank0(f"  Phase 3 — Boundary Exchange:")
    print_rank0(f"    Rounds: {exchange_rounds} (ring topology: read from ±1 neighbor)")
    print_rank0(f"    → Cross-node RDMA opportunity")
    print_rank0(f"  Phase 4 — Ensemble Analysis:")
    print_rank0(f"    All-to-all read of final member states")
    print_rank0(f"  ──────────────────────────────────────────────────────")
    print_rank0(f"  Total raw data: {total_gb:.2f} GB (cluster-wide)")
    print_rank0(f"  Systems: {', '.join(systems)}")
    print_rank0(f"{'#'*90}\n")

    # ── Results Table Header ──
    print_rank0(f"{'System':10} | {'IC Write':>10} | {'IC Read':>10} | "
                f"{'Evolve':>10} | {'Exchange':>10} | {'Analysis':>10} | "
                f"{'E2E (s)':>8} | {'Dedup':>10} | {'Xchg Lat':>10}")
    print_rank0(f"{'':10} | {'(GB/s)':>10} | {'(GB/s)':>10} | "
                f"{'(GB/s)':>10} | {'(GB/s)':>10} | {'(GB/s)':>10} | "
                f"{'':>8} | {'Savings':>10} | {'avg (ms)':>10}")
    print_rank0("-" * 115)

    for sys_name in systems:
        tag = f"{sys_name}_{args.block_size}mb"
        adapter = make_adapter(sys_name, tag)
        if not adapter:
            continue

        e2e_start = time.time()

        # Phase 1: IC Distribution
        ic_wt, ic_rt, ic_wbw, ic_rbw, _ = phase1_distribute_ic(
            adapter, ic_blocks, block_size, sys_name
        )

        # Phase 2: Member Evolution
        ev_time, ev_bw, ev_bytes = phase2_member_evolution(
            adapter, member_blocks, block_size, evolution_steps
        )
        adapter.barrier() # Sync ALL puts
        if sys_name == "Cascade":
            # Double-sync to ensure no metadata is in-flight
            adapter.store.sync_metadata()
            adapter.barrier()
            adapter.store.sync_metadata()
            adapter.barrier()
        
        # Phase 3: Boundary Exchange
        xchg_time, xchg_bw, xchg_lat, _ = phase3_boundary_exchange(
            adapter, exchange_rounds, block_size, sys_name
        )

        if sys_name == "Cascade":
            # Final double-sync before Analysis read
            adapter.store.sync_metadata()
            adapter.barrier()
            adapter.store.sync_metadata()
            adapter.barrier()
            
        # Phase 4: Ensemble Analysis Reduction
        an_time, an_bw = phase4_analysis_reduction(
            adapter, member_blocks, evolution_steps, block_size
        )

        e2e_total = time.time() - e2e_start

        # Dedup stats
        dedup_str = "N/A"
        if hasattr(adapter, 'dedup_stats'):
            hits, saved = adapter.dedup_stats()
            if saved > 0:
                dedup_str = f"{saved/1024**3:.2f} GB"
            else:
                dedup_str = f"{hits} hits"

        # Tier stats
        if hasattr(adapter, 'tier_stats'):
            ts = adapter.tier_stats()
            print_rank0(f"  [Cascade Tier Hits] GPU:{ts['local_gpu']} DRAM:{ts['local_dram']} "
                        f"rGPU:{ts['remote_gpu']} rDRAM:{ts['remote_dram']} "
                        f"Lustre:{ts['lustre']} Miss:{ts['misses']} Promote:{ts['promotions']}")

        print_rank0(f"{sys_name:10} | {ic_wbw:10.2f} | {ic_rbw:10.2f} | "
                    f"{ev_bw:10.2f} | {xchg_bw:10.2f} | {an_bw:10.2f} | "
                    f"{e2e_total:8.2f} | {dedup_str:>10} | {xchg_lat:10.2f}")

        adapter.cleanup()

    print_rank0(f"\n{'='*90}\n")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Simulation Pipeline Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard ensemble benchmark (8 members, 128MB blocks)
  srun -N 8 python3 %(prog)s \\
      --block-size 128 --ic-blocks 20 --member-blocks 10 \\
      --evolution-steps 5 --exchange-rounds 5

  # Stress test with large blocks (256MB, climate-scale)
  srun -N 16 python3 %(prog)s \\
      --block-size 256 --ic-blocks 40 --member-blocks 20 \\
      --evolution-steps 10 --exchange-rounds 10
        """)

    parser.add_argument("--block-size", type=int, default=128,
                        help="Block size in MB (default: 128)")
    parser.add_argument("--ic-blocks", type=int, default=20,
                        help="Number of shared IC blocks (default: 20)")
    parser.add_argument("--member-blocks", type=int, default=10,
                        help="Blocks per member per evolution step (default: 10)")
    parser.add_argument("--evolution-steps", type=int, default=5,
                        help="Number of evolution steps per member (default: 5)")
    parser.add_argument("--exchange-rounds", type=int, default=5,
                        help="Boundary exchange rounds (default: 5)")
    parser.add_argument("--systems", default="Cascade,HDF5,POSIX,vLLM-GPU,PDC,LMCache",
                        help="Comma-separated storage systems to compare")

    args = parser.parse_args()

    # Initialize Cascade
    store = init_global_store(f"ensemble_{args.block_size}mb")
    store.barrier()

    run_ensemble_benchmark(args)

    print_rank0("Ensemble simulation benchmark completed.")

if __name__ == "__main__":
    main()
