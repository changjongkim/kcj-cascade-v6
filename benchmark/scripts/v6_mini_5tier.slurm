#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -J v6_mini_5tier
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/mini_5tier_%j.out
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -e
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "================================================================================"
echo " Starting Mini 5-Tier Verification (Loop: 2, 4, 8 nodes)"
echo "================================================================================"

for N in 2 4 8; do
    RANKS=$((N * 4))
    echo "--- Testing N=$N nodes ($RANKS ranks) ---"
    
    srun -N $N -n $RANKS --gpus-per-node=4 \
        python3 benchmark/scripts/v6_mini_5tier.py 2>&1 | tee benchmark/logs/v6_mini_5tier_N${N}.log

    # Cleanup Lustre tmp files between runs
    rm -rf benchmark/tmp/lustre_mini_r* 2>/dev/null || true
done

