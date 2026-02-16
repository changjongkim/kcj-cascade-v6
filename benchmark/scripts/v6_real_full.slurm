#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_real_full
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/real_full_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/real_full_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj

module reset 2>/dev/null || true
module load PrgEnv-gnu gcc-native/12.3 cray-mpich cudatoolkit/12.4 craype-accel-nvidia80

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
export PYTHONPATH=$CASCADE_BUILD_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/cray/pe/gcc/12.2.0/snos/lib64:/opt/cray/pe/lib64:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1

PYTHON_BIN=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/python3

# Install missing dependencies if needed
$PYTHON_BIN -m pip install h5py redis -q || echo "Pip install failed, trying to continue..."

# Create tmp directory for baselines
mkdir -p benchmark/tmp

# Start Redis for the Redis baseline on ALL nodes
srun -N 4 -n 4 --ntasks-per-node=1 redis-server --port 16379 --save "" --appendonly no &
REDIS_PID=$!
sleep 10

echo "###################################################################"
echo " Real Data Full Comparison (Cascade vs Baselines)"
echo "###################################################################"

SYSTEMS="Cascade,LMCache,HDF5"

for N in 1 2 4; do
    BLOCKS=$((160 / N))
    [ $BLOCKS -lt 1 ] && BLOCKS=1
    echo "--- Scaling Test N=$N nodes ($N ranks), $BLOCKS blocks/rank ---"
    srun -N $N -n $N --gpus-per-node=4 \
        $PYTHON_BIN benchmark/scripts/v6_real_comparison.py $SYSTEMS --blocks-per-rank $BLOCKS
done

# Cleanup Redis
kill $REDIS_PID

echo "###################################################################"
echo " FULL REAL DATA TESTS COMPLETE"
echo "###################################################################"
