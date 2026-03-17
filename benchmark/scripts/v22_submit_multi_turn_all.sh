#!/bin/bash
SYSTEMS=("cascade" "lmcache" "pdc" "vllm-gpu" "hdf5" "redis")
for sys in "${SYSTEMS[@]}"; do
    sbatch benchmark/scripts/v22_multi_turn_8n_${sys}.slurm
done
