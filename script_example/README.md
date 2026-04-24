# CASCADE — Example Scripts

This directory contains representative Slurm scripts and Python drivers used
to produce the results in the CASCADE paper. Each subdirectory maps to a
specific AD task ($T_i$) and paper figure/table.

## Layout

| Directory | AD Task | Paper Element | Purpose |
|---|---|---|---|
| `00_setup/` | $T_1$, $T_3$ | — | Conda environment + C++/CUDA backend build |
| `01_throughput_scalability/` | $T_4$ | Figure 6 | Weak & strong scaling (1–64 nodes) for CASCADE vs 5 baselines |
| `02_tail_latency_burst/` | $T_5$ | Figure 7, Table 2 | P99.9 tail latency CDF + bursty traffic resilience |
| `03_tier_latency/` | $T_6$ | Figure 8 | Retrieval latency across GPU HBM / DRAM / Lustre tiers |
| `04_variable_blocks/` | $T_7$ | Figure 9 | Stability under non-uniform block sizes (log-normal) |
| `05_sensitivity/` | $T_8$ | Figures 10, 11, 12 | Prefix sharing, memory over-subscription, deduplication |
| `06_e2e_inference/` | $T_9$ | Figure 13 | End-to-end inference with CASCADE vs vLLM variants |
| `07_deepcam/` | $T_{10}$ | Figure 14 | MLPerf HPC DeepCAM reconfigurability (Original / No-Dedup / Streaming) |

## Usage

1. Adapt Slurm directives (`#SBATCH --account`, `--constraint`, `--qos`) to your cluster.
2. Source `00_setup/setup_env.sh`, then run `00_setup/build_cpp.sh`.
3. Generate benchmark traces (see the paper's inference_benchmark/workload.py).
4. Submit experiments via `sbatch <script>` or the provided `submit_*.sh`.

## Baselines covered

- **LMCache** — Disk backend (Lustre) and Redis backend (128GB centralized)
- **HDF5** — Independent I/O
- **PDC** — Object-centric data management
- **vLLM** — APC / LMCache-backed modes
- **CASCADE** — Our system

## Notes

- Scripts in this directory are representative subsets. The full per-node
  sweep (1/2/4/8/16/32/64 nodes) lives in `benchmark/scripts/` and
  `benchmark/slurm/Nn/` of the CASCADE repository.
- Output parsing and figure generation live in `paper/scripts/` and
  `cascade_paper/Figures/` of the CASCADE repository.
- DeepCAM uses the MLPerf HPC benchmark suite
  (https://github.com/mlcommons/hpc/tree/main/deepcam); the 512GB dataset
  must be staged separately per MLCommons instructions.
