#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J cascade_weak_8n
#SBATCH -o vllm_integration/logs/weak_scaling_8n_%j.out
#SBATCH -e vllm_integration/logs/weak_scaling_8n_%j.err

echo "=========================================="
echo "Cascade-vLLM Weak Scaling: 8 Nodes"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "==========================================" module load python/3.11
module load cudatoolkit/12.2
module load <mpi-module>

cd ${REPO_ROOT}
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

export PYTHONPATH=${REPO_ROOT}:$PYTHONPATH
export CASCADE_ROOT=${REPO_ROOT}

STORAGE_PATH="${REPO_ROOT}/vllm_integration/cascade_store_weak_${SLURM_JOB_ID}"
export VLLM_CASCADE_LUSTRE_PATH=$STORAGE_PATH

mkdir -p $STORAGE_PATH
mkdir -p vllm_integration/logs

echo ""
echo "Configuration:"
echo " Storage Path: $STORAGE_PATH"
echo " Nodes: 8"
echo " Requests: 64 (8 per node)"
echo "" echo "=========================================="
echo "N=8 (64 requests, 160MB blocks)"
echo "=========================================="
srun --overlap -N 8 -n 8 \
  python3 script_example/06_e2e_inference/weak_scaling_driver.py \
  --num-requests 8 \
  --block-size-mb 160 \
  --storage-path "${STORAGE_PATH}" \
  --gpu-capacity-gb 32 \
  --shm-capacity-gb 64 \
  --enable-dedup \
  --enable-compression \
  --enable-prefetch \
  --run-id "${SLURM_JOB_ID}_N8"

echo ""
echo "=========================================="
echo " 8-Node Test Complete"
echo "=========================================="
echo "Results saved to: ${STORAGE_PATH}" exit 0
