#!/usr/bin/env python3
"""
visualize_sec20_30.py
Generates all figures for README Sections 20-30.
Run: python3 benchmark/scripts/visualize_sec20_30.py
Output: benchmark/figures/fig_sec{N}*.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
    "figure.dpi": 150,
})

COLORS = {
    "Cascade":      "#E74C3C",   # red
    "LMCache-Disk": "#3498DB",   # blue
    "LMCache-Redis":"#9B59B6",   # purple
    "PDC":          "#27AE60",   # green
    "LLM-GPU":      "#F39C12",   # orange
    "HDF5-Indep":   "#1ABC9C",   # teal
}
MARKERS = {"Cascade":"o","LMCache-Disk":"s","LMCache-Redis":"D","PDC":"^","LLM-GPU":"v","HDF5-Indep":"P"}
NODES_64 = [1, 2, 4, 8, 16, 32, 64]
NODES_8  = [1, 2, 4, 8]

def savefig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")

# =============================================================================
# Fig 1 — Section 20: V10 Cluster-Scale Scalability (Weak + Strong)
# =============================================================================
def fig_sec20():
    nodes = NODES_64[:-1]  # up to 32N
    data_weak_ttft = {
        "Cascade":       [12.3, 38.3, 37.2, 55.4, 53.7, 49.9],
        "LMCache-Disk":  [46.9, 213.4, 214.2, 214.1, 214.2, 214.9],
        "LMCache-Redis": [205.9, 200.9, 204.9, 206.7, 215.8, 206.6],
        "PDC":           [49.6, 213.5, 217.7, 211.4, 214.4, 216.6],
        "LLM-GPU":       [68.3, 234.5, 236.4, 232.3, 241.2, 231.0],
        "HDF5-Indep":    [80.0, 243.9, 270.1, 189.4, 194.1, 204.6],
    }
    data_weak_tput = {
        "Cascade":       [74.2, 51.7, 104.1, 141.9, 293.6, 640.5],
        "LMCache-Disk":  [21.3, 9.4, 18.7, 37.4, 74.7, 148.9],
        "LMCache-Redis": [4.9, 10.0, 19.5, 38.7, 74.2, 154.9],
        "PDC":           [20.1, 9.4, 18.4, 37.8, 74.6, 147.7],
        "LLM-GPU":       [14.6, 8.5, 16.9, 34.4, 66.3, 138.5],
        "HDF5-Indep":    [12.5, 8.2, 14.8, 42.2, 82.4, 156.4],
    }
    data_strong_ttft = {
        "Cascade":       [13.0, 33.8, 33.1, 32.5, 37.9, 46.4],
        "LMCache-Disk":  [46.2, 209.8, 209.7, 207.8, 213.3, 214.8],
        "LMCache-Redis": [209.8, 203.3, 205.9, 207.3, 206.8, 207.3],
        "PDC":           [46.3, 210.8, 206.5, 209.8, 214.4, 211.0],
        "LLM-GPU":       [126.7, 230.9, 226.6, 232.4, 238.6, 231.2],
        "HDF5-Indep":    [77.0, 275.9, 260.6, 271.8, 240.9, 188.3],
    }
    data_strong_tput = {
        "Cascade":       [75.9, 59.1, 120.6, 244.2, 419.8, 685.2],
        "LMCache-Disk":  [21.7, 9.5, 19.1, 38.5, 75.0, 148.9],
        "LMCache-Redis": [4.8, 9.8, 19.4, 38.6, 77.4, 154.5],
        "PDC":           [21.6, 9.5, 19.4, 38.1, 74.6, 151.6],
        "LLM-GPU":       [7.9, 8.7, 17.6, 34.4, 67.0, 138.4],
        "HDF5-Indep":    [13.0, 7.2, 15.3, 29.4, 66.4, 169.9],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Section 20: V10 Cluster-Scale Scalability (Llama-2 160MB blocks)", fontsize=13, fontweight="bold")

    datasets = [
        (axes[0,0], data_weak_ttft,   "Weak Scaling — TTFT (ms)", True),
        (axes[0,1], data_weak_tput,   "Weak Scaling — Throughput (req/s)", False),
        (axes[1,0], data_strong_ttft, "Strong Scaling — TTFT (ms, 128 req fixed)", True),
        (axes[1,1], data_strong_tput, "Strong Scaling — Throughput (req/s)", False),
    ]
    for ax, data, title, lower_better in datasets:
        for sys, vals in data.items():
            lw = 2.5 if sys == "Cascade" else 1.2
            ax.plot(nodes, vals, marker=MARKERS[sys], color=COLORS[sys],
                    label=sys, linewidth=lw, markersize=6 if sys=="Cascade" else 5)
        ax.set_xticks(nodes); ax.set_xticklabels([f"{n}N" for n in nodes])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Nodes")
        if lower_better:
            ax.set_ylabel("TTFT (ms)")
        else:
            ax.set_ylabel("Throughput (req/s)")

    handles = [mpatches.Patch(color=COLORS[s], label=s) for s in COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0,0.06,1,1])
    savefig(fig, "fig_sec20_v10_scalability.png")


# =============================================================================
# Fig 2 — Section 21: Qwen 320MB Weak + Strong Scaling
# =============================================================================
def fig_sec21():
    nodes = NODES_64
    data_weak_ttft = {
        "Cascade":       [33.68, 74.83, 84.09, 100.82, 82.69, 80.91, 86.55],
        "LMCache-Disk":  [92.43, 407.19, 413.33, 412.82, None, None, None],
        "LMCache-Redis": [398.46, 395.21, 393.16, 388.62, 386.06, 392.35, 390.51],
        "PDC":           [89.76, 412.61, 411.75, 414.66, 412.81, 412.27, 412.18],
        "LLM-GPU":       [132.27, 445.96, 450.58, 451.80, 449.76, 449.35, 448.36],
        "HDF5-Indep":    [191.47, 513.71, 570.40, 636.75, 957.56, 1523.00, 3097.17],
    }
    data_weak_tput = {
        "Cascade":       [29.65, 26.43, 49.26, 87.23, 202.19, 407.32, 771.76],
        "LMCache-Disk":  [10.81, 4.91, 9.68, 19.38, None, None, None],
        "LMCache-Redis": [2.51, 5.07, 10.19, 20.60, 41.48, 81.64, 163.99],
        "PDC":           [11.13, 4.85, 9.71, 19.29, 38.76, 77.62, 155.29],
        "LLM-GPU":       [7.56, 4.48, 8.88, 17.71, 35.58, 71.21, 142.75],
        "HDF5-Indep":    [5.22, 3.89, 7.15, 13.09, 18.88, 26.98, 30.42],
    }
    data_strong_ttft = {
        "Cascade":       [68.14, 68.20, 83.50, 65.94, 86.35, 102.26, 56.47],
        "LMCache-Disk":  [87.69, 406.41, 412.67, 404.53, None, None, None],
        "LMCache-Redis": [369.30, 388.95, 393.39, 388.51, 387.34, 394.07, 389.85],
        "PDC":           [91.20, 314.00, 403.17, 409.19, 409.58, 407.67, 411.24],
        "LLM-GPU":       [305.94, 475.77, 446.47, 449.41, 453.54, 450.97, 452.62],
        "HDF5-Indep":    [189.03, 506.99, 592.23, 707.35, 1041.89, 1524.88, 2816.45],
    }
    data_strong_tput = {
        "Cascade":       [14.67, 29.33, 54.39, 180.94, 491.59, 719.54, 1072.15],
        "LMCache-Disk":  [11.40, 4.92, 9.69, 19.78, None, None, None],
        "LMCache-Redis": [2.71, 5.14, 10.17, 20.60, 41.35, 81.27, 164.39],
        "PDC":           [10.96, 6.37, 9.92, 19.55, 39.06, 78.49, 155.61],
        "LLM-GPU":       [3.27, 4.21, 8.96, 17.80, 35.28, 70.95, 141.35],
        "HDF5-Indep":    [5.29, 3.94, 6.84, 11.88, 17.52, 28.07, 38.44],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Section 21: Qwen-2.5-72B Scaling Benchmarks (320MB blocks)", fontsize=13, fontweight="bold")

    datasets = [
        (axes[0,0], data_weak_ttft,   "Weak Scaling — TTFT (ms)", "TTFT (ms)"),
        (axes[0,1], data_weak_tput,   "Weak Scaling — Throughput (req/s)", "req/s"),
        (axes[1,0], data_strong_ttft, "Strong Scaling — TTFT (ms, 128 req fixed)", "TTFT (ms)"),
        (axes[1,1], data_strong_tput, "Strong Scaling — Throughput (req/s)", "req/s"),
    ]
    for ax, data, title, ylabel in datasets:
        for sys, vals in data.items():
            x = [nodes[i] for i, v in enumerate(vals) if v is not None]
            y = [v for v in vals if v is not None]
            lw = 2.5 if sys == "Cascade" else 1.2
            ax.plot(x, y, marker=MARKERS[sys], color=COLORS[sys],
                    label=sys, linewidth=lw, markersize=6 if sys=="Cascade" else 4)
        ax.set_xscale("log", base=2); ax.set_xticks(nodes)
        ax.set_xticklabels([f"{n}N" for n in nodes])
        ax.set_title(title, fontsize=10); ax.set_xlabel("Nodes"); ax.set_ylabel(ylabel)

    handles = [mpatches.Patch(color=COLORS[s], label=s) for s in COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0,0.06,1,1])
    savefig(fig, "fig_sec21_qwen_scalability.png")


# =============================================================================
# Fig 3 — Section 23: Hot/Warm/Cold Recovery at 8N, 16N, 32N
# =============================================================================
def fig_sec23():
    systems = ["Cascade", "PDC", "LMCache-Disk", "LLM-GPU", "HDF5-Indep", "LMCache-Redis"]
    tiers   = ["HOT", "WARM", "COLD"]

    lat_8N = {
        "Cascade":       [16.52, 15.32, 15.41],
        "PDC":           [47.10, 55.35, 155.24],
        "LMCache-Disk":  [48.20, 56.81, 144.93],
        "LLM-GPU":       [77.35, 77.01, 76.75],
        "HDF5-Indep":    [189.70, 86.78, 189.89],
        "LMCache-Redis": [239.28, 406.93, 213.51],
    }
    lat_16N = {
        "Cascade":       [13.78, 12.77, 12.86],
        "PDC":           [47.66, 56.22, 155.87],
        "LMCache-Disk":  [46.90, 55.33, 55.11],
        "LLM-GPU":       [77.01, 77.02, 77.26],
        "HDF5-Indep":    [190.34, 85.85, 187.67],
        "LMCache-Redis": [898.37, 740.42, 209.49],
    }
    lat_32N = {
        "Cascade":       [10.47, 9.48, 9.58],
        "PDC":           [47.73, 55.91, 157.32],
        "LMCache-Disk":  [46.88, 54.67, 134.84],
        "LLM-GPU":       [77.16, 77.01, 62.80],
        "HDF5-Indep":    [101.89, 109.88, 225.13],
        "LMCache-Redis": [None, None, None],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("Section 23: Hot/Warm/Cold Recovery Latency (160MB block)", fontsize=13, fontweight="bold")

    for ax, data, title in zip(axes, [lat_8N, lat_16N, lat_32N], ["8 Nodes", "16 Nodes", "32 Nodes"]):
        x = np.arange(len(tiers))
        w = 0.12
        offsets = np.linspace(-(len(systems)-1)*w/2, (len(systems)-1)*w/2, len(systems))
        for i, sys in enumerate(systems):
            vals = [v if v is not None else 0 for v in data[sys]]
            bars = ax.bar(x + offsets[i], vals, width=w*0.9, color=COLORS[sys],
                         label=sys, alpha=0.88)
            if sys == "Cascade":
                for bar, val in zip(bars, data[sys]):
                    if val:
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                                f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                                fontweight="bold", color=COLORS["Cascade"])
        ax.set_xticks(x); ax.set_xticklabels(tiers, fontsize=11)
        ax.set_ylabel("Latency (ms)"); ax.set_title(title, fontsize=11)

    handles = [mpatches.Patch(color=COLORS[s], label=s) for s in systems]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0,0.09,1,1])
    savefig(fig, "fig_sec23_hot_warm_cold.png")


# =============================================================================
# Fig 4 — Section 24: Semantic Eviction Stability
# =============================================================================
def fig_sec24():
    nodes = NODES_8
    data = {
        "Cascade":       [98.44, 12.81, 14.40, 13.22],
        "LMCache-Disk":  [97.12, 104.59, 87.18, 48.77],
        "PDC":           [126.60, 92.75, 134.37, 121.77],
        "LLM-GPU":       [142.01, 107.88, 143.41, 127.49],
        "HDF5-Indep":    [168.87, 191.97, 201.01, 199.30],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Section 24: Semantic Eviction Stability\n(1.5× GPU Oversubscription, Important Prefix TTFT)", fontsize=11, fontweight="bold")
    for sys, vals in data.items():
        lw = 2.5 if sys == "Cascade" else 1.2
        ax.plot(nodes, vals, marker=MARKERS[sys], color=COLORS[sys],
                label=sys, linewidth=lw, markersize=7 if sys == "Cascade" else 5)
    ax.fill_between(nodes, data["Cascade"], alpha=0.12, color=COLORS["Cascade"])
    ax.set_xticks(nodes); ax.set_xticklabels([f"{n}N" for n in nodes])
    ax.set_ylabel("TTFT of Protected Prefix (ms)"); ax.set_xlabel("Nodes")
    ax.legend(); ax.set_title("Lower is better")
    ax.text(0.5, -0.15, "💡 Cascade's Semantic Eviction protects critical prefix blocks even under 1.5× oversubscription.\n"
            "Redis: LOST (0% retention). Baselines: 90–200ms from Lustre fallback.",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    savefig(fig, "fig_sec24_semantic_eviction.png")


# =============================================================================
# Fig 5 — Section 25: Tail Latency (P50/P99/P99.9) across scales
# =============================================================================
def fig_sec25():
    nodes = [1, 2, 4, 8, 16, 32]
    tail_p99 = {
        "Cascade":       [13.00, 83.77, 48.21, 48.22, 76.13, 48.81],
        "HDF5-Indep":    [146.48, 249.49, 251.96, 247.86, 253.62, 248.07],
        "LMCache-Disk":  [60.34, 214.55, 218.27, 218.69, 218.52, 218.59],
        "PDC":           [58.63, 217.69, 217.58, 218.00, 219.37, 219.08],
        "LLM-GPU":       [86.99, 239.72, 471.06, 466.25, 996.44, 603.04],
        "LMCache-Redis": [227.02, 236.16, 267.04, 622.07, None, None],
    }
    tail_p999 = {
        "Cascade":       [17.44, 88.64, 50.14, 50.95, 87.32, 52.79],
        "HDF5-Indep":    [156.44, 3205.58, 4914.96, 13973.19, 44111.95, 43953.64],
        "LMCache-Disk":  [66.07, 217.55, 227.06, 229.37, 230.68, 239.22],
        "PDC":           [59.61, 227.12, 241.15, 228.97, 282.67, 252.02],
        "LLM-GPU":       [93.00, 244.02, 1095.30, 994.90, 1337.46, 1122.73],
        "LMCache-Redis": [260.39, 248.64, 292.55, 679.00, None, None],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Section 25: Tail Latency Distribution Analysis (500 requests concurrent)", fontsize=12, fontweight="bold")

    for ax, data, ylabel in zip(axes, [tail_p99, tail_p999], ["P99 TTFT (ms)", "P99.9 TTFT (ms)"]):
        for sys, vals in data.items():
            x = [nodes[i] for i, v in enumerate(vals) if v is not None]
            y = [v for v in vals if v is not None]
            lw = 2.5 if sys == "Cascade" else 1.2
            ax.plot(x, y, marker=MARKERS[sys], color=COLORS[sys],
                    label=sys, linewidth=lw, markersize=6)
        ax.set_yscale("log"); ax.set_xticks(nodes)
        ax.set_xticklabels([f"{n}N" for n in nodes])
        ax.set_ylabel(ylabel); ax.set_xlabel("Nodes")
        ax.set_title(ylabel + " (log scale)")

    handles = [mpatches.Patch(color=COLORS[s], label=s) for s in COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0,0.08,1,1])
    savefig(fig, "fig_sec25_tail_latency.png")


# =============================================================================
# Fig 6 — Section 27: Mixed R/W YCSB @ 8N and 16N
# =============================================================================
def fig_sec27():
    systems  = ["Cascade", "PDC", "LMCache-Disk", "LLM-GPU", "HDF5-Indep", "REDIS"]
    workloads = ["Workload A\n(95R/5W)", "Workload B\n(50R/50W)", "Workload C\n(Scan)"]

    ops_8N = {
        "Cascade":      [3288.7, 376.0, 53427.5],
        "PDC":          [1010.5, 289.2, 11700.0],
        "LMCache-Disk": [1048.9, 317.7, 2179.0],
        "HDF5-Indep":   [649.8,  229.5, 2249.9],
        "REDIS":        [216.3,  123.7, 226.5],
        "LLM-GPU":      [1222.6, 309.8, 11525.4],
    }
    ops_16N = {
        "Cascade":      [6162.8, 703.0, 185042.3],
        "PDC":          [1326.2, 590.5, 35827.4],
        "LMCache-Disk": [1416.6, 588.4, 4524.4],
        "HDF5-Indep":   [463.3,  305.0, 733.4],
        "REDIS":        [200.6,  132.1, 215.6],
        "LLM-GPU":      [1223.9, 627.5, 25631.1],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Section 27: Mixed Read/Write YCSB Throughput (ops/sec)", fontsize=12, fontweight="bold")

    for ax, data, title in zip(axes, [ops_8N, ops_16N], ["8 Nodes (32 GPUs)", "16 Nodes (64 GPUs)"]):
        x = np.arange(len(workloads))
        w = 0.12
        n = len(systems)
        offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)
        for i, sys in enumerate(systems):
            color = COLORS.get(sys, "#95A5A6")
            ax.bar(x + offsets[i], data[sys], width=w*0.9, color=color,
                   label=sys, alpha=0.88)
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(workloads, fontsize=9)
        ax.set_ylabel("Ops/sec (log scale)"); ax.set_title(title)

    handles = [mpatches.Patch(color=COLORS.get(s, "#95A5A6"), label=s) for s in systems]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0,0.09,1,1])
    savefig(fig, "fig_sec27_ycsb_mixed.png")


# =============================================================================
# Fig 7 — Section 28: Index Lookup Scalability
# =============================================================================
def fig_sec28():
    blocks  = [1_000, 10_000, 100_000, 500_000]
    xlabels = ["1K", "10K", "100K", "500K"]
    data = {
        "Cascade":       [0.05, 0.02, 0.04, 0.04],
        "LMCache-Disk":  [4.30, 7.54, 4.52, 8.18],
        "LLM-GPU":       [3.67, 2.92, 2.39, 2.10],
        "PDC":           [2.41, 1.83, 1.86, 2.04],
        "LMCache-Redis": [0.43, 0.25, 0.25, 0.29],
        "HDF5-Indep":    [19.03, 24.25, 51.25, 82.75],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Section 28: Index Lookup P99 Latency vs. Block Count\n(O(1) sharded hash vs. file system / B-tree)", fontsize=11, fontweight="bold")
    for sys, vals in data.items():
        lw = 2.5 if sys == "Cascade" else 1.2
        ax.plot(range(len(blocks)), vals, marker=MARKERS.get(sys,"o"),
                color=COLORS.get(sys,"#95A5A6"), label=sys, linewidth=lw)
    ax.set_xticks(range(len(blocks))); ax.set_xticklabels(xlabels)
    ax.set_xlabel("Number of Stored Blocks"); ax.set_ylabel("P99 Lookup Latency (ms)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig_sec28_index_lookup.png")


# =============================================================================
# Fig 8 — Section 29.2: Single-Node Index Scalability (Disk systems comparison)
# =============================================================================
def fig_sec29_2():
    systems = ["Cascade\n(Disk-COLD)", "Cascade\n(Disk-HOT)", "LMCache", "PDC", "HDF5-Indep", "LMCache-Redis"]
    bw      = [6.76, 7.03, 0.94, 0.95, 0.93, 8.21]
    p50     = [0.02, 0.02, 18.30, 17.92, 17.12, 0.06]
    colors  = [COLORS["Cascade"], COLORS["Cascade"], COLORS["LMCache-Disk"],
               COLORS["PDC"], COLORS["HDF5-Indep"], COLORS["LMCache-Redis"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Section 29.2: Single-Node Index Scalability (50K Blocks @ 800GB)\nAll disk-backed or overflow-to-disk mode", fontsize=11, fontweight="bold")

    x = np.arange(len(systems))
    bars1 = ax1.bar(x, bw, color=colors, alpha=0.88, edgecolor="white", linewidth=1)
    ax1.set_xticks(x); ax1.set_xticklabels(systems, fontsize=8, rotation=15)
    ax1.set_ylabel("Aggregate Bandwidth (GB/s)"); ax1.set_title("Bandwidth (higher is better)")
    for bar, v in zip(bars1, bw):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{v:.2f}", ha="center", fontsize=9)

    bars2 = ax2.bar(x, p50, color=colors, alpha=0.88, edgecolor="white", linewidth=1)
    ax2.set_xticks(x); ax2.set_xticklabels(systems, fontsize=8, rotation=15)
    ax2.set_ylabel("P50 TTFT (ms)"); ax2.set_title("P50 Latency (lower is better)")
    for bar, v in zip(bars2, p50):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"{v:.2f}", ha="center", fontsize=9)

    ax1.set_yticks(np.arange(0, 10, 1))
    fig.tight_layout()
    savefig(fig, "fig_sec29_2_singlenode_disk.png")


# =============================================================================
# Fig 9 — Section 30: Time Breakdown (per-tier cost + realistic model)
# =============================================================================
def fig_sec30():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Section 30: Cascade get() Time Breakdown — Qwen 320MB Blocks", fontsize=12, fontweight="bold")

    # ── 30a: Per-tier BW by node count ──
    ax = axes[0]
    nodes_bw    = [1, 2, 4, 8]
    local_bw    = [11.27, 16.23, 12.46, 16.23]
    remote_bw   = [None, 5.41, 2.32, 5.39]
    lustre_ref  = 0.93

    ax.plot(nodes_bw, local_bw, "o-", color=COLORS["Cascade"], lw=2.5, ms=8, label="Local GPU")
    remote_x = [n for n, v in zip(nodes_bw, remote_bw) if v is not None]
    remote_y = [v for v in remote_bw if v is not None]
    ax.plot(remote_x, remote_y, "s--", color="#E67E22", lw=2, ms=7, label="Remote RDMA")
    ax.axhline(lustre_ref, color="#7F8C8D", linestyle=":", lw=1.5, label=f"Lustre baseline ({lustre_ref} GB/s)")
    ax.set_xticks(nodes_bw); ax.set_xticklabels([f"{n}N" for n in nodes_bw])
    ax.set_ylabel("Bandwidth (GB/s)"); ax.set_xlabel("Nodes")
    ax.set_title("Per-tier Bandwidth\n(isolated measurements)")
    ax.legend(fontsize=8)

    # ── 30b: Phase breakdown % (pie-like bar, always 100% data transfer) ──
    ax = axes[1]
    phases = ["Index\nLookup", "Data\nTransfer", "Python\nDeser"]
    pcts   = [0.004, 99.992, 0.004]
    bcolors = ["#3498DB", COLORS["Cascade"], "#27AE60"]
    bars = ax.bar(phases, pcts, color=bcolors,  alpha=0.88)
    for bar, p in zip(bars, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{p:.3f}%", ha="center", fontsize=9)
    ax.set_ylabel("% of E2E Time (Local Read)"); ax.set_ylim(0, 115)
    ax.set_title("Time Breakdown Composition\n(Local Read, any node count)")
    ax.text(0, 50, "~1 μs", ha="center", va="center", fontsize=8, color="white")
    ax.text(1, 50, "~19,000 μs", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(2, 50, "~0.5 μs", ha="center", va="center", fontsize=8, color="white")

    # ── 30c: Realistic E2E model — stacked bar ──
    ax = axes[2]
    nodes = [1, 2, 4, 8]
    p_local  = [1.0, 0.5, 0.25, 0.125]
    p_remote = [0.0, 0.5, 0.75, 0.875]
    T_local  = 19.3
    T_remote = 58.0

    local_ms  = [p * T_local  for p in p_local]
    remote_ms = [p * T_remote for p in p_remote]
    index_ms  = [0.001] * 4  # negligible

    x = np.arange(len(nodes))
    ax.bar(x, local_ms,  label="Local GPU",    color=COLORS["Cascade"],  alpha=0.88)
    ax.bar(x, remote_ms, bottom=local_ms,      label="Remote RDMA",      color="#E67E22",  alpha=0.88)
    ax.bar(x, index_ms,  bottom=[l+r for l,r in zip(local_ms, remote_ms)], label="Index+Python", color="#BDC3C7", alpha=0.88)

    total = [l+r+i for l,r,i in zip(local_ms, remote_ms, index_ms)]
    for i, (t, loc_p, rem_p) in enumerate(zip(total, p_local, p_remote)):
        ax.text(i, t+1, f"{t:.0f}ms", ha="center", fontsize=9, fontweight="bold")
        if rem_p > 0:
            ax.text(i, local_ms[i] + remote_ms[i]/2, f"RDMA\n{rem_p*100:.0f}%",
                    ha="center", va="center", fontsize=8, color="white")

    ax.set_xticks(x); ax.set_xticklabels([f"{n}N" for n in nodes])
    ax.set_ylabel("Expected E2E Latency (ms)"); ax.set_xlabel("Nodes")
    ax.set_title("Realistic E2E Model\n(without Locality Promotion)\nP(remote)=(N-1)/N")
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig_sec30_time_breakdown.png")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating Sections 20-30 figures...")
    fig_sec20()
    fig_sec21()
    fig_sec23()
    fig_sec24()
    fig_sec25()
    fig_sec27()
    fig_sec28()
    fig_sec29_2()
    fig_sec30()
    print(f"\n✅ All figures saved to {FIG_DIR}/")
