#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J v6_lustre_cold
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/lustre_cold_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/logs/lustre_cold_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
module reset 2>/dev/null || true
module load PrgEnv-gnu gcc-native/12.3 cray-mpich cudatoolkit/12.4 craype-accel-nvidia80

export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp
export PYTHONPATH=$CASCADE_BUILD_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/cray/pe/gcc/12.2.0/snos/lib64:/opt/cray/pe/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/lib64:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1

PYTHON_BIN=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/python3

mkdir -p benchmark/tmp

echo "###################################################################"
echo " Lustre Cold Storage Benchmark Scaling (1, 2, 4, 8 nodes)"
echo "###################################################################"

SYSTEMS="PDC,vLLM-GPU,HDF5,LMCache,Cascade"

for N in 1 2 4 8; do
    echo ">>>> Running Lustre Cold Test: N=$N nodes <<<<"
    srun -N $N -n $N --gpus-per-node=4 \
        $PYTHON_BIN benchmark/scripts/v6_lustre_cold_bench.py --systems $SYSTEMS --block-size-mb 512 --num-blocks 10
done
