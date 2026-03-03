#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J recovery_hdf5
#SBATCH -o benchmark/logs/recovery_hdf5_%j.out
#SBATCH -e benchmark/logs/recovery_hdf5_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

# OFI Settings
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=32768
export FI_CXI_OFLOW_BUF_SIZE=2097152
export FI_CXI_DEFAULT_VNI=0

SYSTEM="hdf5"
BLOCK_MB=160

echo "=== Recovery Profiling: ${SYSTEM} ==="
for i in 8; do
    echo ""
    echo "▶ SCALE: N=${i} nodes"
    srun --overlap -N $i -n $i --ntasks-per-node=1 \
        python3 benchmark/scripts/v12_recovery_profiling.py \
        --system $SYSTEM \
        --block-size-mb $BLOCK_MB \
        --num-blocks 10
    sleep 5
done

echo "✅ Done"
