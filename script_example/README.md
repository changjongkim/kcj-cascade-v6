# CASCADE — Example Scripts

Self-contained Slurm scripts, drivers, adapters, data generators, and
plotting scripts for reproducing CASCADE paper results. Each
subdirectory maps to an AD task ($T_i$) and paper figure/table.

## Layout

| Directory | Task | Paper | Slurm | Driver | Coverage |
|---|---|---|---:|---|---|
| `00_setup/` | $T_1$, $T_3$ | — | 0 | `setup_env.sh`, `build_cpp.sh` | env + build |
| `01_throughput_scalability/` | $T_4$ | Fig 6 | 10 | `throughput_driver.py` | 5 systems × {small example, 64n full} |
| `02_tail_latency_burst/` | $T_5$ | Fig 7, Table 2 | 9 | `tail_driver.py`, `burst_driver.py` | tail at 1/8/32/64n (multi-system) + burst at 32n × 5 systems |
| `03_tier_latency/` | $T_6$ | Fig 8 | 11 | `tier_driver.py` | 4 systems × {8n, 64n} + tier microbenchmarks (gpu/dram/lustre) |
| `04_variable_blocks/` | $T_7$ | Fig 9 | 5 | `variable_block_driver.py` | 5 systems × 16n |
| `05_sensitivity/prefix/` | $T_8$ | Fig 10 | 6 | `throughput_driver.py`, `dedup_prefix_driver.py` | 5 systems + dedup |
| `05_sensitivity/oversubscription/` | $T_8$ | Fig 11 | 1 | `oversubscription_driver.py` | 5 systems × 1×–16× |
| `05_sensitivity/dedup/` | $T_8$ | Fig 12 | 5 | `dedup_driver.py` | 5 systems × 8n |
| `06_e2e_inference/` | $T_9$ | Fig 13 | 8 | 4 drivers + `cascade_vllm_engine.py` | CASCADE + vLLM baseline × {1,2,4,8}n |
| `07_deepcam/` | $T_{10}$ | Fig 14 | 8 | 4 drivers | 3 modes × {16n, 64n} + baselines |
| `benchmark/` | shared | — | — | `run_benchmark.py`, `config.py`, `adapters/`, data modules | CASCADE, HDF5, LMCache-Disk, LMCache-Redis, PDC |
| `data_generation/` | trace prep | — | — | `data_generator.py`, `data_generator_real.py`, `dataset_loader.py` | ShareGPT/OpenOrca/CNN-DailyMail trace preprocessing |
| `analysis/` | plotting | — | — | `generate_eval_figures.py`, `generate_eval_figures_detailed.py` | Figure 6–14 generation from result CSVs |

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

## Reproduction Workflow

1. **Setup** ($T_1$, $T_3$): `bash 00_setup/setup_env.sh && bash 00_setup/build_cpp.sh`
2. **Trace generation** ($T_2$): `python3 data_generation/data_generator.py` (see the script for dataset-specific flags)
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
4. **Plotting**: `python3 analysis/generate_eval_figures.py <result-dir>`

## Baselines

- LMCache-Disk — LMCache with Lustre file backend
- LMCache-Redis — LMCache with centralized 128GB Redis backend
- HDF5 — parallel I/O with independent mode
- PDC — object-centric data management
- vLLM — APC and LMCache-backed modes (end-to-end inference only;
  invoked directly, not through the shared adapter layer)
- CASCADE — this work

## Notes

- DeepCAM requires the MLPerf HPC benchmark dataset
  (https://github.com/mlcommons/hpc/tree/main/deepcam); stage the
  512GB dataset per MLCommons instructions.
- `03_tier_latency/{tier_gpu,tier_dram,tier_lustre}.slurm` are
  CASCADE-only tier microbenchmarks; Figure 8 comparisons use
  `tier_{system}_{8n,64n}.slurm` which reuse `throughput_driver.py`
  with `--tier-mode {hot,warm,cold}`.
- `01_throughput_scalability/*_full.slurm` cover the full 1–64 node
  weak/strong sweep (regular queue, 1 hr). Short debug variants
  (`*.slurm` without `_full` suffix) use 1–4 nodes.
