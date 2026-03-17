#!/bin/bash
SYSTEMS=("cascade" "lmcache" "pdc" "vllm-gpu" "hdf5" "redis")
for sys in "${SYSTEMS[@]}"; do
    sbatch benchmark/scripts/v20_var_32n_${sys}.slurm
done
