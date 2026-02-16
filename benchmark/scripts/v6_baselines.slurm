#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_baselines
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/baselines_%j.out

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj

# Fast initialization - skip conda to avoid timeout
module reset 2>/dev/null || true
module load PrgEnv-gnu gcc-native/12.3 cray-mpich cudatoolkit/12.4 craype-accel-nvidia80

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
export PYTHONPATH=$CASCADE_BUILD_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/cray/pe/gcc/12.2.0/snos/lib64:/opt/cray/pe/lib64:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1

# Use system Python with mpi4py
PYTHON_BIN=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/python3
BENCH_SCRIPT=benchmark/scripts/v6_real_comparison.py

echo "================================================================================"
echo " 📚 BASELINES COMPARISON (HDF5, LMCache, Redis, PDC)"
echo "================================================================================"

for N in 1 2 4 8; do
    RANKS=$((N * 4))
    BLOCKS_PER_RANK=$((640 / RANKS))
    [ $BLOCKS_PER_RANK -lt 1 ] && BLOCKS_PER_RANK=1

    echo "--- Testing Baselines N=$N nodes ($RANKS ranks) ---"
    
    # Run others (excluding Cascade)
    srun -N $N -n $RANKS --gpus-per-node=4 \
        $PYTHON_BIN $BENCH_SCRIPT LMCache,HDF5,Redis,PDC --blocks-per-rank $BLOCKS_PER_RANK
    
    rm -rf benchmark/tmp/hdf5_comp_r* benchmark/tmp/pdc_comp_r* benchmark/tmp/lmcache_comp_r* 2>/dev/null || true
done
