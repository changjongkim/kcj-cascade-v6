#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_real_debug
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/real_debug_%j.out

set -e
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
BENCH_SCRIPT=benchmark/scripts/v6_real_comparison.py

echo "================================================================================"
echo " Cascade V6 - MINI REAL-DATA COMPARISON (Debug Mode, ~100GB Total)"
echo " Job: $SLURM_JOB_ID | Start: $(date)"
echo " Systems: Cascade, LMCache, HDF5, Redis, PDC"
echo "================================================================================"

for N in 1 2 4 8; do
    RANKS=$((N * 4))
    # 100GB total target:
    # N=1: 640 blocks/node (19 blocks per rank) -> approx 100GB total
    # But for a scaling study, we usually keep work-per-node or total fix.
    # Let's target 100GB total at N=8.
    BLOCKS_PER_RANK=$((640 / RANKS))
    [ $BLOCKS_PER_RANK -lt 1 ] && BLOCKS_PER_RANK=1

    echo "--- Scaling to N=$N nodes ($RANKS ranks), Blocks/Rank: $BLOCKS_PER_RANK ---"
    
    srun -N $N -n $RANKS --gpus-per-node=4 \
        python3 $BENCH_SCRIPT Cascade,LMCache,HDF5,Redis,PDC --blocks-per-rank $BLOCKS_PER_RANK 2>&1 | tee benchmark/logs/real_debug_N${N}.log
    
    # Cleanup between node counts
    rm -rf benchmark/tmp/lustre_comp_r* benchmark/tmp/hdf5_comp_r* benchmark/tmp/pdc_comp_r* 2>/dev/null || true
done

echo "================================================================================"
echo " Completed: $(date)"
echo "================================================================================"
