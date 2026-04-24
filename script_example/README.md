# CASCADE — Example Scripts

Self-contained Slurm scripts and Python drivers for reproducing the
CASCADE paper results. Each subdirectory maps to an AD task ($T_i$)
and paper figure/table.

## Layout

| Directory | Task | Paper | Contents |
|---|---|---|---|
| `00_setup/` | $T_1$, $T_3$ | — | `setup_env.sh`, `build_cpp.sh` |
| `01_throughput_scalability/` | $T_4$ | Fig 6 | 6 Slurm (CASCADE + 5 baselines) + `throughput_driver.py` |
| `02_tail_latency_burst/` | $T_5$ | Fig 7, Table 2 | 7 Slurm (tail + burst) + `tail_driver.py`, `burst_driver.py` |
| `03_tier_latency/` | $T_6$ | Fig 8 | GPU/DRAM tier Slurm + `tier_driver.py` |
| `04_variable_blocks/` | $T_7$ | Fig 9 | 5 Slurm + `variable_block_driver.py` |
| `05_sensitivity/prefix/` | $T_8$ | Fig 10 | Prefix Slurm + `throughput_driver.py`, `dedup_prefix_driver.py` |
| `05_sensitivity/oversubscription/` | $T_8$ | Fig 11 | `oversubscription_driver.py` |
| `05_sensitivity/dedup/` | $T_8$ | Fig 12 | 6 Slurm + `dedup_driver.py` |
| `06_e2e_inference/` | $T_9$ | Fig 13 | 5 Slurm + `cascade_e2e_driver.py`, `cascade_strong_driver.py`, `vllm_baseline_driver.py`, `cascade_vllm_engine.py` |
| `07_deepcam/` | $T_{10}$ | Fig 14 | 8 Slurm + `original_driver.py`, `nodedup_driver.py`, `streaming_driver.py`, `deepcam_driver.py` |

## Usage

1. Adapt the Slurm header (`#SBATCH --account`, `-C`, `-q`) to your cluster.
2. Run `source 00_setup/setup_env.sh` then `bash 00_setup/build_cpp.sh`.
3. Submit each experiment with `sbatch <script>.slurm` or the provided
   `submit_*.sh` wrappers in the corresponding directory.

## Baselines

- **LMCache** — Disk (Lustre) and Redis (centralized 128GB) backends
- **HDF5** — parallel I/O with independent mode
- **PDC** — object-centric data management
- **vLLM** — APC and LMCache-backed modes
- **CASCADE** — this work

## Notes

- Each Slurm `cd`s to the CASCADE repository root, then invokes the
  driver via `script_example/<task>/<driver>.py`. Imports resolve via
  `PYTHONPATH` set in `setup_env.sh`.
- Subsets shown here are representative (e.g., 8n/16n/64n). Full
  per-node sweeps are available under `benchmark/scripts/` and
  `benchmark/slurm/` of the main repository.
- DeepCAM requires the MLPerf HPC benchmark dataset
  (https://github.com/mlcommons/hpc/tree/main/deepcam); stage the
  512GB dataset per MLCommons instructions.
