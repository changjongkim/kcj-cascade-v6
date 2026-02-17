#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_qwen_scaling
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/qwen_scaling_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/qwen_scaling_%j.err

set -ex
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

PYTHON_BIN=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/python3
SCRIPT=benchmark/scripts/v6_contention_scaling_all.py

echo "###################################################################"
echo " REAL-WORLD SCALING: LLaMA-3 vs Qwen2.5"
echo "###################################################################"

# Only test 8 nodes for quick verification in debug queue
N=8
SYSTEMS="Cascade,LMCache" # Focus on Cascade vs LMCache comparison

for MODEL in llama-3-70b qwen-2.5-72b qwen-2.5-32b qwen-2.5-7b; do
    echo ""
    echo ">>>> Running Realistic Scaling: Model=$MODEL, N=$N Nodes <<<<"
    # Use real-world data distribution (Shared prefix + local)
    srun -N $N -n $N --gpus-per-node=4 \
        $PYTHON_BIN $SCRIPT --mode weak --data-type real --model $MODEL --systems $SYSTEMS
done
