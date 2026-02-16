#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -J v6_strong_5tier
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/strong_5tier_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/strong_5tier_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -e
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
BENCH_SCRIPT=benchmark/scripts/v6_5tier_bench.py

echo "================================================================================"
echo " 👑 Cascade V6 - 5-TIER STRONG SCALING STUDY (1.0 TB Fixed)"
echo " Job: $SLURM_JOB_ID | Start: $(date)"
echo "================================================================================"

for N in 1 2 4 8; do
    RANKS=$((N * 4))
    echo "--- Testing N=$N nodes ($RANKS ranks), Target: 1024 GB Total Fixed ---"

    srun -N $N -n $RANKS --gpus-per-node=4 \
        $PYTHON_BIN $BENCH_SCRIPT --mode strong --total-gb 1024.0

    rm -rf benchmark/tmp/lustre_5tier_r* 2>/dev/null || true
done
