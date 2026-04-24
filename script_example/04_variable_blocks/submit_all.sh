#!/bin/bash

echo "Submitting Variable Block Size Experiments to SLURM..."

sbatch benchmark/scripts/v20_variable_block_8n.slurm

sbatch benchmark/scripts/v20_variable_block_32n.slurm

echo "8-Node and 32-Node Variable block experiments submitted."
