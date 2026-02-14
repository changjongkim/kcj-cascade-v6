#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -J v6_comparison
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/comparison_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/comparison_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -e
export MPICH_GPU_SUPPORT_ENABLED=1

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
mkdir -p benchmark/tmp benchmark/logs

# Load environment
module load cudatoolkit
module load python
# Ensure redis/HDF5 dependencies are met
pip install redis h5py --user --quiet 2>/dev/null || true

SCRIPTS_DIR=benchmark/scripts

echo "================================================================================"
echo " 🚀 Cascade V6 Comparison Scaling Study (1, 2, 4, 8 Nodes)"
echo " Start: $(date)"
echo "================================================================================"

# Systems to test
SYSTEMS="Cascade,LMCache,HDF5,Redis,PDC"

for N in 1 2 4 8; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo " 🌏 Running on N=$N Nodes ($((N*4)) ranks)"
    echo "--------------------------------------------------------------------------------"
    
    # Start Redis servers on each node (Cascade style distributed management)
    # We use a trick: srun per node to start redis-server in background
    srun -N $N -n $N --ntasks-per-node=1 bash -c "pkill -u $USER redis-server || true; /pscratch/sd/s/sgkim/kcj/Cascade-kcj/third_party/redis/src/redis-server --port 16379 --daemonize yes"
    sleep 2
    
    # Run the benchmark
    srun -N $N -n $((N*4)) python3 $SCRIPTS_DIR/v6_comparison_multi_system.py $SYSTEMS
    
    # Clean up redis
    srun -N $N -n $N --ntasks-per-node=1 pkill -u $USER redis-server || true
done

echo ""
echo "================================================================================"
echo " ✅ Comparison Scaling Study Complete: $(date)"
echo "================================================================================"
