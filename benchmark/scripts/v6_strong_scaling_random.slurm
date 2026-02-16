#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -J v6_rand_read
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/strong_rand_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/strong_rand_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -e

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
mkdir -p benchmark/logs benchmark/tmp

# Load environment using the robust script
source setup_env.sh

# Explicitly set build dir
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
BENCH_SCRIPT=benchmark/scripts/v6_strong_scaling_bench_random.py
LOG_DIR=benchmark/logs

echo "================================================================================"
echo " 🔬 Cascade V6 STRONG SCALING [RANDOM READ 500GB] Study"
echo " Job: $SLURM_JOB_ID | Start: $(date)"
echo "================================================================================"

for N in 1 2 4 8; do
    RANKS=$((N * 4))
    LOG_FILE=${LOG_DIR}/v6_strong_rand_N${N}_${SLURM_JOB_ID}.log

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  N=${N} nodes, ${RANKS} ranks | Target: 500GB Total | Random Read"
    echo "────────────────────────────────────────────────────────────────────"

    srun -N $N -n $RANKS --gpus-per-node=4 \
        python3 $BENCH_SCRIPT 2>&1 | tee $LOG_FILE

    # Clean up tmp files between runs
    rm -rf benchmark/tmp/lustre_strong_rand_r* 2>/dev/null || true
done

echo ""
echo "================================================================================"
echo " ✅ Random Read Study Complete: $(date)"
echo "================================================================================"
