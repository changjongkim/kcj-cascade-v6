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

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
mkdir -p benchmark/tmp benchmark/logs

# Load environment using the robust script
source setup_env.sh

SCRIPTS_DIR=benchmark/scripts

echo "================================================================================"
echo " 🚀 Cascade V6 Comparison Scaling Study (1, 2, 4, 8 Nodes)"
echo " Start: $(date)"
echo "================================================================================"

# Systems to test (Redis removed due to setup complexity)
SYSTEMS="Cascade,LMCache,HDF5,PDC"

for N in 1 2 4 8; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo " 🌏 Running on N=$N Nodes ($((N*4)) ranks)"
    echo "--------------------------------------------------------------------------------"
    
    # Run the benchmark
    srun -N $N -n $((N*4)) python3 $SCRIPTS_DIR/v6_comparison_multi_system.py $SYSTEMS
    
done

echo ""
echo "================================================================================"
echo " ✅ Comparison Scaling Study Complete: $(date)"
echo "================================================================================"
