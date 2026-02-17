#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_qwen_only
#SBATCH -o benchmark/logs/qwen_only_%j.out
#SBATCH -e benchmark/logs/qwen_only_%j.err

# Setup Environment
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

PYTHON_BIN=$(which python3)
chmod +x benchmark/scripts/v6_contention_scaling_all.py

echo "###################################################################"
echo " REAL-WORLD SCALING: Qwen 2.5 Series (Cascade Only)"
echo "###################################################################"

# Configuration
N=8
SYSTEMS="Cascade"  # Skipping LMCache to avoid I/O errors observed previously
MODELS="qwen-2.5-72b qwen-2.5-32b qwen-2.5-7b"

for MODEL in $MODELS; do
    echo ""
    echo ">>>> Running Realistic Scaling: Model=$MODEL, N=$N Nodes <<<<"
    
    # Run Benchmark
    srun -N $N -n $N --gpus-per-node=4 \
        $PYTHON_BIN benchmark/scripts/v6_contention_scaling_all.py \
        --mode weak \
        --data-type real \
        --model $MODEL \
        --systems $SYSTEMS
        
    echo "----------------------------------------------------------------"
done

echo "Benchmark Suite Completed."
