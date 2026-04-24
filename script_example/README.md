# CASCADE — Example Scripts

Self-contained Slurm scripts, drivers, and adapters for reproducing
CASCADE paper results. Each subdirectory maps to an AD task ($T_i$)
and paper figure/table.

## Layout

| Directory | Task | Paper |
|---|---|---|
| `00_setup/` | $T_1$, $T_2$, $T_3$ | — |
| `01_throughput_scalability/` | $T_4$ | Fig 6 |
| `02_tail_latency_burst/` | $T_5$ | Fig 7, Table 2 |
| `03_tier_latency/` | $T_6$ | Fig 8 |
| `04_variable_blocks/` | $T_7$ | Fig 9 |
| `05_sensitivity/prefix/` | $T_8$ | Fig 10 |
| `05_sensitivity/oversubscription/` | $T_8$ | Fig 11 |
| `05_sensitivity/dedup/` | $T_8$ | Fig 12 |
| `06_e2e_inference/` | $T_9$ | Fig 13 |
| `07_deepcam/` | $T_{10}$ | Fig 14 |
| `benchmark/` | shared | — |

## Environment

Set before submission:

```
export REPO_ROOT=/path/to/cascade
export SCRATCH=/your/scratch
```

Then:

```
bash 00_setup/setup_env.sh
bash 00_setup/build_cpp.sh
```

Replace `#SBATCH -A <account>` with the reviewer's allocation.

Each Slurm automatically prepends `${REPO_ROOT}/script_example` to
`PYTHONPATH`, so drivers' `from benchmark.run_benchmark import
get_adapter` resolves to the adapters shipped in this directory.

## Workflow

1. **Setup** ($T_1$, $T_3$): `bash 00_setup/setup_env.sh && bash 00_setup/build_cpp.sh`
2. **Trace generation** ($T_2$): `python3 00_setup/generate_traces.py` (see the script for dataset-specific flags)
3. **Per-figure reproduction**:
   - Fig 6: `sbatch 01_throughput_scalability/cascade_full.slurm` (and baselines)
   - Fig 7: `sbatch 02_tail_latency_burst/tail_{1,8,32,64}n.slurm`
   - Table 2: `bash 02_tail_latency_burst/submit_burst_all.sh`
   - Fig 8: `sbatch 03_tier_latency/tier_{cascade,lmcache,hdf5,pdc}_{8n,64n}.slurm`
   - Fig 9: `bash 04_variable_blocks/submit_all.sh`
   - Fig 10: `sbatch 05_sensitivity/prefix/{cascade,lmcache,hdf5,pdc,redis}_prefix.slurm`
   - Fig 11: `sbatch 05_sensitivity/oversubscription/oversubscription_8n.slurm`
   - Fig 12: `sbatch 05_sensitivity/dedup/{cascade,hdf5,lmcache,pdc,redis}_8n.slurm`
   - Fig 13: `sbatch 06_e2e_inference/{cascade,vllm_baseline}_{1,2,4,8}n.slurm`
   - Fig 14: `sbatch 07_deepcam/{original,nodedup,streaming}_{16n,64n}.slurm`

## Baselines

- LMCache-Disk — LMCache with Lustre file backend
- LMCache-Redis — LMCache with centralized 128GB Redis backend
- HDF5 — parallel I/O with independent mode
- PDC — object-centric data management
- vLLM — APC and LMCache-backed modes (end-to-end inference only;
  invoked directly, not through the shared adapter layer)
- CASCADE

## Notes

- DeepCAM requires the MLPerf HPC benchmark dataset
  (https://github.com/mlcommons/hpc/tree/main/deepcam); stage the
  512GB dataset per MLCommons instructions.
- To test a different node count, edit the Slurm file directly
  (`#SBATCH -N` and any per-script node list).
