#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_strong
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/v6_strong_scaling_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/v6_strong_scaling_%j.out
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -e
export MPICH_GPU_SUPPORT_ENABLED=1

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
mkdir -p benchmark/logs benchmark/tmp

# Load env
source cascade_Code/cpp/build_mpi/load_env.sh 2>/dev/null || {
    module reset 2>/dev/null || true
    module load gcc-native/12.3
    module load cudatoolkit/12.4
    module load cray-mpich/8.1.30
    export CRAY_ACCEL_TARGET=nvidia80
    export MPICH_GPU_SUPPORT_ENABLED=1
}

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_mpi
BENCH_SCRIPT=benchmark/scripts/v6_strong_scaling_bench.py
LOG_DIR=benchmark/logs

echo "================================================================================"
echo " 🔬 Cascade V6 STRONG SCALING Study (Fixed Data = 80K blocks ≈ 12.5 GB)"
echo " Job: $SLURM_JOB_ID | Start: $(date)"
echo "================================================================================"

for N in 1 2 4 8; do
    RANKS=$((N * 4))
    LOG_FILE=${LOG_DIR}/v6_strong_N${N}_${SLURM_JOB_ID}.log

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  N=${N} nodes, ${RANKS} ranks"
    echo "────────────────────────────────────────────────────────────────────"

    srun -N $N -n $RANKS --gpus-per-node=4 \
        python3 $BENCH_SCRIPT 2>&1 | tee $LOG_FILE

    # Clean up tmp files between runs
    rm -rf benchmark/tmp/lustre_strong_r* 2>/dev/null || true
done

echo ""
echo "================================================================================"
echo " ✅ Strong Scaling Study Complete: $(date)"
echo "================================================================================"

# Print summary table
echo ""
echo "========== STRONG SCALING SUMMARY =========="
for N in 1 2 4 8; do
    LOG_FILE=${LOG_DIR}/v6_strong_N${N}_${SLURM_JOB_ID}.log
    echo "--- N=${N} ---"
    grep -E 'Write Time|Write Throughput|Read Time|Read Throughput|Total Data' $LOG_FILE 2>/dev/null || echo "  (no data)"
done
echo "============================================="
