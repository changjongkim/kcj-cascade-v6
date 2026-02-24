#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_app_checkpoint
#SBATCH -o benchmark/logs/app_checkpoint_%j.out
#SBATCH -e benchmark/logs/app_checkpoint_%j.err

# ============================================================
# App 1: Scientific Checkpoint/Restart Benchmark
# ============================================================
# Simulates iterative CFD/Climate solver checkpoint I/O:
#   - 5 physical fields (T, P, Vx, Vy, Vz) × 256 MB each
#   - 20 timesteps, checkpoint every 5 → 4 checkpoints
#   - 5% incremental delta per timestep → 95% dedup potential
#   - Strong scaling: fixed problem per rank
#
# Expected output:
#   - Cascade dedup saves ~90% of storage
#   - HDF5/POSIX write everything (no dedup) → higher I/O volume
#   - Restart read: Cascade from GPU/DRAM tier, others from Lustre
# ============================================================

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
mkdir -p benchmark/logs benchmark/tmp

echo "###################################################################"
echo " SCIENTIFIC APP 1: Iterative Solver Checkpoint/Restart"
echo " Nodes: $SLURM_NNODES | GPUs: $((SLURM_NNODES * 4))"
echo " Field: 256 MB × 5 fields = 1.25 GB/checkpoint/rank"
echo " Timeline: 20 timesteps, ckpt every 5 = 4 checkpoints"
echo " Delta: 5% per timestep → ~95% dedup opportunity"
echo "###################################################################"

# ── Main Benchmark ──
srun python3 benchmark/scripts/v6_app_checkpoint.py \
    --mode checkpoint \
    --field-size 256 \
    --num-fields 5 \
    --timesteps 20 \
    --ckpt-interval 5 \
    --delta-pct 5 \
    --systems Cascade,HDF5,POSIX

echo ""
echo "###################################################################"
echo " DELTA SENSITIVITY SWEEP"
echo " Varying change rate: 1%, 2%, 5%, 10%, 25%, 50%, 100%"
echo "###################################################################"

# ── Delta Sweep (shows dedup vs change rate) ──
srun python3 benchmark/scripts/v6_app_checkpoint.py \
    --mode sweep \
    --field-size 128 \
    --num-fields 3 \
    --systems Cascade,HDF5,POSIX

echo "Checkpoint benchmark completed."
