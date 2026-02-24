#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_app_ensemble
#SBATCH -o benchmark/logs/app_ensemble_%j.out
#SBATCH -e benchmark/logs/app_ensemble_%j.err

# ============================================================
# App 2: Ensemble Simulation Pipeline Benchmark
# ============================================================
# Simulates weather/climate ensemble forecast I/O workflow:
#
# Phase 1 — Shared IC Distribution:
#   Rank 0 writes initial condition → all ranks read the SAME data.
#   Cascade: dedup recognizes identical data → single copy + RDMA reads.
#   Baselines: each rank reads from Lustre → metadata contention.
#
# Phase 2 — Member Evolution:
#   Each ranks writes unique member state (per-member data).
#   Tests independent write throughput at scale.
#
# Phase 3 — Boundary Exchange (Halo Exchange):
#   Ring topology: each rank reads neighbor's boundary data.
#   Cascade: uses RDMA Tier 3/4 directly.
#   Baselines: go through Lustre → much higher latency.
#
# Phase 4 — Analysis Reduction:
#   All ranks read ALL members' final state (all-to-all).
#   Tests cross-node read scalability.
# ============================================================

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
mkdir -p benchmark/logs benchmark/tmp

echo "###################################################################"
echo " SCIENTIFIC APP 2: Ensemble Simulation Pipeline"
echo " Nodes: $SLURM_NNODES | Members: $SLURM_NNODES (1 per node)"
echo " Block: 128 MB"
echo " IC: 20 shared blocks (2.5 GB) × ${SLURM_NNODES} readers"
echo " Evolution: 10 blocks × 5 steps per member"
echo " Exchange: 5 rounds (ring topology)"
echo " Analysis: All-to-all final state read"
echo "###################################################################"

srun python3 benchmark/scripts/v6_app_ensemble.py \
    --block-size 128 \
    --ic-blocks 20 \
    --member-blocks 10 \
    --evolution-steps 5 \
    --exchange-rounds 5 \
    --systems Cascade,HDF5,POSIX

echo "Ensemble benchmark completed."
