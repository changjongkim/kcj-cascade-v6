#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/slingshot_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/slingshot_%j.err
#SBATCH -J slingshot_test

# =============================================================================
# SLINGSHOT-11 BANDWIDTH TEST
# =============================================================================
# 1 rank per node → GUARANTEE cross-node communication
# Large messages for bandwidth saturation
# =============================================================================

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade

module load python/3.11
module load cudatoolkit
module load cray-mpich

cd $PROJECT_DIR

JOB_ID=$SLURM_JOB_ID
echo "Job: $JOB_ID, Nodes: $SLURM_NNODES, Ranks: $SLURM_NTASKS"

srun python3 << 'PYTHON_SCRIPT'
import os, sys, time, json, numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NPROCS = comm.Get_size()
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '.')

if RANK == 0:
    print("="*75)
    print("SLINGSHOT-11 CROSS-NODE BANDWIDTH TEST")
    print("="*75)
    print(f"1 rank per node → GUARANTEED cross-node MPI")
    print(f"Nodes: {NPROCS}")
    print("")
    sys.stdout.flush()

comm.Barrier()

# Test multiple message sizes
msg_sizes = [
    1 * 1024 * 1024,    # 1MB
    16 * 1024 * 1024,   # 16MB
    64 * 1024 * 1024,   # 64MB
    256 * 1024 * 1024,  # 256MB
    512 * 1024 * 1024,  # 512MB
]

results = {}

for msg_size in msg_sizes:
    # Ring exchange: 0→1→2→3→0
    send_to = (RANK + 1) % NPROCS
    recv_from = (RANK - 1 + NPROCS) % NPROCS
    
    data = np.random.bytes(msg_size)
    recv_buf = bytearray(msg_size)
    
    comm.Barrier()
    t0 = time.perf_counter()
    
    # Non-blocking for simultaneous send/recv
    req_send = comm.Isend([np.frombuffer(data, dtype=np.uint8), MPI.BYTE], dest=send_to, tag=0)
    req_recv = comm.Irecv([recv_buf, MPI.BYTE], source=recv_from, tag=0)
    
    MPI.Request.Waitall([req_send, req_recv])
    
    comm.Barrier()
    elapsed = time.perf_counter() - t0
    
    # Each node sends AND receives msg_size bytes
    bw_gbps = 2 * msg_size / elapsed / (1024**3)
    
    # Aggregate
    total_bw = comm.reduce(bw_gbps, op=MPI.SUM, root=0)
    
    if RANK == 0:
        avg_bw = total_bw / NPROCS
        size_mb = msg_size / (1024**2)
        print(f"  {size_mb:6.0f}MB: {avg_bw:.2f} GB/s per link, {total_bw:.1f} GB/s aggregate")
        results[f"{int(size_mb)}MB"] = {"per_link": avg_bw, "aggregate": total_bw}
        sys.stdout.flush()

comm.Barrier()

# Also test all-to-all (simulates global address space lookups)
if RANK == 0:
    print("")
    print(">>> ALL-TO-ALL BANDWIDTH (Global Address Space simulation)")
    sys.stdout.flush()

msg_size = 64 * 1024 * 1024  # 64MB per rank
send_buf = np.random.bytes(msg_size * NPROCS)
recv_buf = bytearray(msg_size * NPROCS)

comm.Barrier()
t0 = time.perf_counter()

# Alltoall - everyone sends to everyone
sendbuf = np.frombuffer(send_buf, dtype=np.uint8).reshape(NPROCS, msg_size)
recvbuf = np.zeros((NPROCS, msg_size), dtype=np.uint8)
comm.Alltoall([sendbuf, MPI.BYTE], [recvbuf, MPI.BYTE])

comm.Barrier()
elapsed = time.perf_counter() - t0

# Total data moved: NPROCS * NPROCS * msg_size (each sends to all)
total_data = NPROCS * NPROCS * msg_size
alltoall_bw = total_data / elapsed / (1024**3)

if RANK == 0:
    print(f"  Alltoall {NPROCS}×{msg_size/1024/1024:.0f}MB: {alltoall_bw:.1f} GB/s total")
    results["alltoall_gbps"] = alltoall_bw
    print("")
    sys.stdout.flush()

# Summary
if RANK == 0:
    print("="*75)
    print("COMPARISON")
    print("="*75)
    print(f"Slingshot-11 cross-node: ~{results['512MB']['per_link']:.1f} GB/s per link")
    print(f"Lustre cold read:        ~17 GB/s aggregate")
    print("")
    
    if results['512MB']['per_link'] > 17:
        speedup = results['512MB']['per_link'] / 17
        print(f"✅ REMOTE DRAM IS {speedup:.1f}x FASTER THAN LUSTRE!")
    else:
        print(f"⚠️  Slingshot slower than expected (theoretical: 100 GB/s)")
    
    print("="*75)
    
    # Save
    out = f"{PROJECT_DIR}/benchmark/results/slingshot_{JOB_ID}.json"
    with open(out, 'w') as f:
        json.dump({"job_id": JOB_ID, "nodes": NPROCS, "results": results}, f, indent=2)
    print(f"Saved: {out}")

PYTHON_SCRIPT

echo "[DONE]"
