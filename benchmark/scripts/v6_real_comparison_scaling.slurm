#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -J v6_real_comp
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/real_comp_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/real_comp_%j.err

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
BENCH_SCRIPT=benchmark/scripts/v6_real_comparison.py

echo "================================================================================"
echo " Cascade V6 - REAL WORKLOAD COMPARISON SCALING STUDY"
echo " Job: $SLURM_JOB_ID | Start: $(date)"
echo " Systems: Cascade, HDF5, PDC"
echo "================================================================================"

# Clear any previous tmp files
rm -rf benchmark/tmp/lustre_comp_r* benchmark/tmp/hdf5_comp_r* benchmark/tmp/pdc_comp_r* 2>/dev/null || true

for N in 1 2 4 8; do
    RANKS=$((N * 4))
    echo "--- Scaling to N=$N nodes ($RANKS ranks) ---"
    
    srun -N $N -n $RANKS --gpus-per-node=4 \
        $PYTHON_BIN $BENCH_SCRIPT Cascade,LMCache,HDF5,Redis,PDC
    
    # Cleanup between node counts
    rm -rf benchmark/tmp/lustre_comp_r* benchmark/tmp/hdf5_comp_r* benchmark/tmp/pdc_comp_r* 2>/dev/null || true
done

echo "================================================================================"
echo " Completed: $(date)"
echo "================================================================================"
