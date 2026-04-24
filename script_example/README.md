# CASCADE — Example Scripts

Self-contained Slurm scripts, drivers, and adapters for reproducing
CASCADE paper results. Each subdirectory maps to an AD task ($T_i$)
and paper figure/table.

## Layout

| Directory | Task | Paper | Contents |
|---|---|---|---|
| `00_setup/` | $T_1$, $T_3$ | — | `setup_env.sh`, `build_cpp.sh` |
| `01_throughput_scalability/` | $T_4$ | Fig 6 | 5 Slurm + `throughput_driver.py` |
| `02_tail_latency_burst/` | $T_5$ | Fig 7, Table 2 | 7 Slurm + `tail_driver.py`, `burst_driver.py` |
| `03_tier_latency/` | $T_6$ | Fig 8 | 2 Slurm + `tier_driver.py` |
| `04_variable_blocks/` | $T_7$ | Fig 9 | 5 Slurm + `variable_block_driver.py` |
| `05_sensitivity/prefix/` | $T_8$ | Fig 10 | 2 Slurm + `throughput_driver.py`, `dedup_prefix_driver.py` |
| `05_sensitivity/oversubscription/` | $T_8$ | Fig 11 | `oversubscription_driver.py` |
| `05_sensitivity/dedup/` | $T_8$ | Fig 12 | 5 Slurm + `dedup_driver.py` |
| `06_e2e_inference/` | $T_9$ | Fig 13 | 5 Slurm + e2e drivers + `cascade_vllm_engine.py` |
| `07_deepcam/` | $T_{10}$ | Fig 14 | 8 Slurm + 4 drivers |
| `benchmark/` | shared | — | `run_benchmark.py`, `config.py`, `adapters/` (CASCADE, HDF5, LMCache, PDC, Redis, vLLM) |

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
sbatch 01_throughput_scalability/cascade.slurm
```

Replace `#SBATCH -A <account>` with the reviewer's allocation.

Each Slurm automatically prepends `${REPO_ROOT}/script_example` to
`PYTHONPATH`, so drivers' `from benchmark.run_benchmark import
get_adapter` resolves to the adapters shipped inside this directory.

## Baselines

- LMCache — Disk (Lustre) and Redis (centralized 128GB) backends
- HDF5 — parallel I/O with independent mode
- PDC — object-centric data management
- vLLM — APC and LMCache-backed modes (used only in end-to-end
  inference and DeepCAM tasks)
- CASCADE — this work

## Notes

- DeepCAM requires the MLPerf HPC benchmark dataset
  (https://github.com/mlcommons/hpc/tree/main/deepcam); stage the
  512GB dataset per MLCommons instructions.
