#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J cascade_weak_4n
#SBATCH -o vllm_integration/logs/weak_scaling_4n_%j.out
#SBATCH -e vllm_integration/logs/weak_scaling_4n_%j.err

echo "=========================================="
echo "Cascade-vLLM Weak Scaling: 4 Nodes"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "=========================================="

module load python/3.11
module load cudatoolkit/12.2
module load cray-mpich

cd ${REPO_ROOT}
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

export PYTHONPATH=${REPO_ROOT}:$PYTHONPATH
export CASCADE_ROOT=${REPO_ROOT}

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20
export MPICH_OFI_NUM_NICS=1

STORAGE_PATH="${REPO_ROOT}/vllm_integration/cascade_store_weak_${SLURM_JOB_ID}"
export VLLM_CASCADE_LUSTRE_PATH=$STORAGE_PATH

mkdir -p $STORAGE_PATH
mkdir -p vllm_integration/logs

echo ""
echo "Configuration:"
echo "Storage Path: $STORAGE_PATH"
echo "Nodes: 4"
echo "Requests: 32 (8 per node)"
echo ""

echo "=========================================="
echo "N=4 (32 requests, 160MB blocks)"
echo "=========================================="
srun --overlap -N 4 -n 4 \
  python3 vllm_integration/cascade_vllm_weak_scaling.py \
  --num-requests 8 \
  --block-size-mb 160 \
  --storage-path "${STORAGE_PATH}"\
  --gpu-capacity-gb 32 \
  --shm-capacity-gb 64 \
  --enable-dedup \
  --enable-compression \
  --enable-prefetch \
  --run-id "${SLURM_JOB_ID}_N4"

echo ""
echo "=========================================="
echo "4-Node Test Complete"
echo "=========================================="
echo "Results saved to: ${STORAGE_PATH}"

exit 0
