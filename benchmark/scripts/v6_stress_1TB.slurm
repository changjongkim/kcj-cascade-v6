#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -J v6_1TB_Stress
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/stress_1TB_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/stress_1TB_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -e

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
mkdir -p benchmark/logs benchmark/tmp

source setup_env.sh

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
BENCH_SCRIPT=benchmark/scripts/v6_stress_1TB.py
LOG_DIR=benchmark/logs

echo "================================================================================"
echo " 🦖 Cascade V6 - 1.0 TERABYTE STRESS TEST"
echo " Job: $SLURM_JOB_ID | Start: $(date)"
echo "================================================================================"

# We test 1, 2, 4, 8 nodes to see the tiering shift
for N in 1 2 4 8; do
    RANKS=$((N * 4))
    LOG_FILE=${LOG_DIR}/v6_stress_1TB_N${N}_${SLURM_JOB_ID}.log

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  N=${N} nodes, ${RANKS} ranks | Target: 1.0 TB TOTAL"
    echo "────────────────────────────────────────────────────────────────────"

    # Important: Increase per-process memory limit if necessary, 
    # but Slurm should handle cgroups based on node allocation.
    srun -N $N -n $RANKS --gpus-per-node=4 \
        python3 $BENCH_SCRIPT 2>&1 | tee $LOG_FILE

    # Clean up Lustre tmp files to free space for next N run
    rm -rf benchmark/tmp/lustre_1TB_r* 2>/dev/null || true
done

echo ""
echo "================================================================================"
echo " ✅ 1.0 TB Stress Test Complete: $(date)"
echo "================================================================================"
