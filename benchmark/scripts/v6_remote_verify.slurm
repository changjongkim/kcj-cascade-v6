#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_remote_verify
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/remote_verify_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/remote_verify_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -e

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
mkdir -p benchmark/logs benchmark/tmp

# Load environment
source setup_env.sh

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "================================================================================"
echo " 🔬 Cascade V6 REMOTE READ VERIFICATION (Tier 3/4)"
echo " Job: $SLURM_JOB_ID | Start: $(date)"
echo " Build: $(ls -la ${CASCADE_BUILD_DIR}/cascade_cpp*.so 2>/dev/null | tail -1)"
echo "================================================================================"

RANKS=$((8 * 4))
LOG_FILE=benchmark/logs/v6_remote_verify_${SLURM_JOB_ID}.log

srun -N 8 -n $RANKS --gpus-per-node=4 \
    python3 benchmark/scripts/v6_remote_read_verify.py 2>&1 | tee $LOG_FILE

# Cleanup
rm -rf benchmark/tmp/lustre_verify_r* 2>/dev/null || true

echo ""
echo "================================================================================"
echo " ✅ Verification Complete: $(date)"
echo "================================================================================"
