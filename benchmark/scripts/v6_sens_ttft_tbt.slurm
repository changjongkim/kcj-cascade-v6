#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J v6_sens_ttft_tbt
#SBATCH -o benchmark/logs/sens_ttft_tbt_%j.out
#SBATCH -e benchmark/logs/sens_ttft_tbt_%j.err

# Setup Environment
cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
SCRIPT=benchmark/scripts/v6_ttft_tbt_benchmark.py
SYSTEMS="Cascade,HDF5,vLLM-GPU,PDC,LMCache"

echo "###################################################################"
echo " SENSITIVITY ANALYSIS: TTFT & TBT (Inference-Level Metrics)"
echo " Fixed: 4 Nodes (16 GPUs), Qwen-2.5-72B"
echo "###################################################################"

# ============================================================
# PART 1: TTFT Sensitivity — Prefix Length
#   Simulates varying context lengths:
#     4 blocks  =  4K tokens  (short prompt)
#    16 blocks  = 16K tokens  (medium context)
#    64 blocks  = 64K tokens  (long context)
#   128 blocks  = 128K tokens (ultra-long context)
# ============================================================

echo ""
echo "========== TTFT: 4 Prefix Blocks (4K tokens, ~1.3 GB) — HOT =========="
srun python3 $SCRIPT --metric ttft --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 4

echo ""
echo "========== TTFT: 16 Prefix Blocks (16K tokens, ~5.1 GB) — HOT =========="
srun python3 $SCRIPT --metric ttft --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 16

echo ""
echo "========== TTFT: 64 Prefix Blocks (64K tokens, ~20.5 GB) — HOT =========="
srun python3 $SCRIPT --metric ttft --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 64

echo ""
echo "========== TTFT: 128 Prefix Blocks (128K tokens, ~41 GB) — HOT =========="
srun python3 $SCRIPT --metric ttft --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 128

# ============================================================
# PART 2: TTFT Cold vs Hot Comparison
#   Same prefix length (16 blocks), comparing cache state impact
# ============================================================

echo ""
echo "========== TTFT: 16 Blocks — COLD START (Lustre → GPU) =========="
srun python3 $SCRIPT --metric ttft --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 16 --cold

echo ""
echo "========== TTFT: 64 Blocks — COLD START (Lustre → GPU) =========="
srun python3 $SCRIPT --metric ttft --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 64 --cold

# ============================================================
# PART 3: TBT Sensitivity — Concurrent Decode Streams
#   Fixed 50 decode steps, varying concurrent request load
# ============================================================

echo ""
echo "========== TBT: 1 Concurrent Stream (50 steps) =========="
srun python3 $SCRIPT --metric tbt --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 16 --decode-steps 50 \
    --concurrent-requests 1

echo ""
echo "========== TBT: 4 Concurrent Streams (50 steps) =========="
srun python3 $SCRIPT --metric tbt --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 16 --decode-steps 50 \
    --concurrent-requests 4

echo ""
echo "========== TBT: 8 Concurrent Streams (50 steps) =========="
srun python3 $SCRIPT --metric tbt --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 16 --decode-steps 50 \
    --concurrent-requests 8

echo ""
echo "========== TBT: 16 Concurrent Streams (50 steps) =========="
srun python3 $SCRIPT --metric tbt --model qwen-2.5-72b \
    --systems $SYSTEMS --prefix-blocks 16 --decode-steps 50 \
    --concurrent-requests 16

echo ""
echo "###################################################################"
echo " ALL TTFT/TBT SENSITIVITY TESTS COMPLETE"
echo "###################################################################"
