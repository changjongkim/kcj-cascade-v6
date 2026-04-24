#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J cascade_vllm_weak_2n
#SBATCH -o vllm_integration/logs/weak_scaling_2n_%j.out
#SBATCH -e vllm_integration/logs/weak_scaling_2n_%j.err

echo "=========================================="
echo "Cascade-vLLM Weak Scaling Benchmark (2N)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "=========================================="

module load python/3.11
module load cudatoolkit/12.2
module load <mpi-module>

cd ${REPO_ROOT}
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

export PYTHONPATH=${REPO_ROOT}:$PYTHONPATH
export CASCADE_ROOT=${REPO_ROOT}

export FI_MR_CACHE_MONITOR=disabled

STORAGE_PATH="${REPO_ROOT}/vllm_integration/cascade_store_weak_${SLURM_JOB_ID}"
export VLLM_CASCADE_LUSTRE_PATH=$STORAGE_PATH

mkdir -p $STORAGE_PATH
mkdir -p vllm_integration/logs

echo ""
echo "Environment Configuration:"
echo "Storage Path: $STORAGE_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

echo "=========================================="
echo "Test 1: N=1 (8 requests, 160MB blocks)"
echo "=========================================="
srun --overlap -N 1 -n 1 --ntasks-per-node=1 \
  python3 vllm_integration/cascade_vllm_weak_scaling.py \
  --num-requests 8 \
  --block-size-mb 160 \
  --storage-path "${STORAGE_PATH}/n1"\
  --gpu-capacity-gb 32 \
  --shm-capacity-gb 64 \
  --enable-dedup \
  --enable-compression \
  --enable-prefetch \
  --run-id "${SLURM_JOB_ID}_N1"

echo ""
echo "Waiting 5 seconds before next test..."
sleep 5

echo "=========================================="
echo "Test 2: N=2 (16 requests, 160MB blocks)"
echo "=========================================="
srun --overlap -N 2 -n 2 --ntasks-per-node=1 \
  python3 vllm_integration/cascade_vllm_weak_scaling.py \
  --num-requests 16 \
  --block-size-mb 160 \
  --storage-path "${STORAGE_PATH}/n2"\
  --gpu-capacity-gb 32 \
  --shm-capacity-gb 64 \
  --enable-dedup \
  --enable-compression \
  --enable-prefetch \
  --run-id "${SLURM_JOB_ID}_N2"

echo ""
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="

echo ""
echo "Results from N=1:"
if [ -f "vllm_integration/results_weak_${SLURM_JOB_ID}_N1.json"]; then
  python3 -c "
import json
with open('vllm_integration/results_weak_${SLURM_JOB_ID}_N1.json') as f:
    data = json.load(f)
    agg = data['aggregate']
    print(f\"Nodes: {agg['nodes']}\")
    print(f\"Total Requests: {agg['total_requests']}\")
    print(f\"Avg Latency: {agg['avg_latency_ms']:.2f} ms\")
    print(f\"P99 Latency: {agg['max_p99_latency_ms']:.2f} ms\")
    print(f\"Throughput: {agg['total_throughput']:.2f} blocks/s\")
"
else
  echo "Results file not found"
fi

echo ""
echo "Results from N=2:"
if [ -f "vllm_integration/results_weak_${SLURM_JOB_ID}_N2.json"]; then
  python3 -c "
import json
with open('vllm_integration/results_weak_${SLURM_JOB_ID}_N2.json') as f:
    data = json.load(f)
    agg = data['aggregate']
    print(f\"Nodes: {agg['nodes']}\")
    print(f\"Total Requests: {agg['total_requests']}\")
    print(f\"Avg Latency: {agg['avg_latency_ms']:.2f} ms\")
    print(f\"P99 Latency: {agg['max_p99_latency_ms']:.2f} ms\")
    print(f\"Throughput: {agg['total_throughput']:.2f} blocks/s\")
"
else
  echo "Results file not found"
fi

echo ""
echo "Weak Scaling Analysis:"
python3 -c "
import json
import sys

try:
    with open('vllm_integration/results_weak_${SLURM_JOB_ID}_N1.json') as f:
        n1 = json.load(f)['aggregate']
    with open('vllm_integration/results_weak_${SLURM_JOB_ID}_N2.json') as f:
        n2 = json.load(f)['aggregate']

    latency_ratio = n2['avg_latency_ms'] / n1['avg_latency_ms']
    throughput_ratio = n2['total_throughput'] / n1['total_throughput']

    print(f\"Latency Ratio (N2/N1): {latency_ratio:.2f}x\")
    print(f\"Throughput Ratio (N2/N1): {throughput_ratio:.2f}x\")
    print(f\"\")

    if latency_ratio < 1.2:
        print(\"Good weak scaling: latency remains stable\")
    elif latency_ratio < 1.5:
        print(\"~ Acceptable weak scaling\")
    else:
        print(\"Poor weak scaling: latency degrades significantly\")

    if throughput_ratio > 1.8:
        print(\"Excellent throughput scaling\")
    elif throughput_ratio > 1.5:
        print(\"~ Good throughput scaling\")
    else:
        print(\"Poor throughput scaling\")

except FileNotFoundError:
    print(\"Could not compute ratios: missing result files\")
    sys.exit(0)
"

echo ""
echo "=========================================="
echo "Storage Usage:"
du -sh $STORAGE_PATH 2>/dev/null || echo "Storage path not created"

echo ""
echo "=========================================="
echo "Weak Scaling Benchmark Completed!"
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo ""
echo "Results saved in:"
echo "- vllm_integration/results_weak_${SLURM_JOB_ID}_N1.json"
echo "- vllm_integration/results_weak_${SLURM_JOB_ID}_N2.json"
echo "=========================================="

rm -rf /dev/shm/cascade_vllm_*

exit 0
