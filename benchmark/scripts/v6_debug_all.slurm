#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_debug_all
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/debug_all_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/debug_all_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj

# Fast initialization - skip conda
module reset 2>/dev/null || true
module load PrgEnv-gnu gcc-native/12.3 cray-mpich cudatoolkit/12.4 craype-accel-nvidia80

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
export PYTHONPATH=$CASCADE_BUILD_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/cray/pe/gcc/12.2.0/snos/lib64:/opt/cray/pe/lib64:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1

PYTHON_BIN=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/python3

echo "###################################################################"
echo " TEST 1: 5-Tier Strong Scaling (12GB total)"
echo " 1 rank/node, each rank manages 4 GPUs + full DRAM"
echo "###################################################################"

for N in 1 2 4; do
    echo "--- Strong Scaling N=$N nodes ($N ranks) ---"
    srun -N $N -n $N --gpus-per-node=4 \
        $PYTHON_BIN benchmark/scripts/v6_5tier_bench.py --mode strong --total-gb 12.0
    rm -rf benchmark/tmp/lustre_5tier_r* 2>/dev/null || true
done

echo "###################################################################"
echo " TEST 2: 5-Tier Weak Scaling (3GB/node)"
echo "###################################################################"

for N in 1 2 4; do
    echo "--- Weak Scaling N=$N nodes ($N ranks) ---"
    srun -N $N -n $N --gpus-per-node=4 \
        $PYTHON_BIN benchmark/scripts/v6_5tier_bench.py --mode weak --node-gb 3.0
    rm -rf benchmark/tmp/lustre_5tier_r* 2>/dev/null || true
done

echo "###################################################################"
echo " TEST 3: Real Data Comparison (Cascade, ~25GB)"
echo "###################################################################"

for N in 1 2 4; do
    BLOCKS=$((160 / N))
    [ $BLOCKS -lt 1 ] && BLOCKS=1
    echo "--- Cascade Real Data N=$N nodes ($N ranks), $BLOCKS blocks/rank ---"
    srun -N $N -n $N --gpus-per-node=4 \
        $PYTHON_BIN benchmark/scripts/v6_real_comparison.py Cascade --blocks-per-rank $BLOCKS
    rm -rf benchmark/tmp/lustre_comp_r* 2>/dev/null || true
done

echo "###################################################################"
echo " ALL TESTS COMPLETE"
echo "###################################################################"
