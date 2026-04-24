#!/bin/bash

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj

echo "=========================================="
echo "Submitting V17 Bursty Traffic Experiments"
echo "=========================================="

mkdir -p benchmark/logs

SYSTEMS="cascade hdf5 lmcache pdc llm_gpu redis"

for sys in $SYSTEMS; do
    script="benchmark/scripts/v17_bursty_${sys}_8n.slurm"
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        sbatch "$script"
        sleep 1
    else
        echo "WARNING: Script not found: $script"
    fi
done

echo "=========================================="
echo "All bursty traffic jobs submitted!"
echo "Check status with: squeue -u \$USER"
echo "=========================================="
