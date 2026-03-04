#!/usr/bin/env python3
"""
visualize_sec20_30.py  —  Sections 20-30 figures for Cascade paper
Run: /global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/python3 benchmark/scripts/visualize_sec20_30.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

FS = 30   # base font
FS_SM = 24
FS_ANN = 22
LW_CAS = 3.5
LW_BASE = 1.8
MS_CAS = 12
MS_BASE = 8

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.35,
    "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "axes.titlesize": FS, "figure.titlesize": FS,
    "legend.fontsize": FS_SM, "legend.framealpha": 0.9,
    "figure.dpi": 150,
})

COLORS = {
    "Cascade":       "#E74C3C",
    "LMCache-Disk":  "#3498DB",
    "LMCache-Redis": "#9B59B6",
    "PDC":           "#27AE60",
    "LLM-GPU":       "#F39C12",
    "HDF5-Indep":    "#1ABC9C",
}
MK = {"Cascade":"o","LMCache-Disk":"s","LMCache-Redis":"D","PDC":"^","LLM-GPU":"v","HDF5-Indep":"P"}

def savefig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")

def xpos(labels):
    """Return evenly-spaced integer positions + label list from node list or label list."""
    return list(range(len(labels))), labels

def line_plot(ax, data, nodes, colors=COLORS, markers=MK):
    """Plot all systems on ax with evenly-spaced x positions."""
    xi, xlabels = xpos([f"{n}N" for n in nodes])
    for sys, vals in data.items():
        px = [xi[i] for i, v in enumerate(vals) if v is not None]
        py = [v for v in vals if v is not None]
        lw = LW_CAS if sys == "Cascade" else LW_BASE
        ms = MS_CAS if sys == "Cascade" else MS_BASE
        ax.plot(px, py, marker=markers.get(sys,"o"), color=colors.get(sys,"#7F8C8D"),
                linewidth=lw, markersize=ms)
    ax.set_xticks(xi); ax.set_xticklabels(xlabels)

def legend_bottom(fig, systems, ncol=3):
    handles = [mpatches.Patch(color=COLORS.get(s,"#7F8C8D"), label=s) for s in systems]
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, 0), fontsize=FS_SM,
               frameon=True, framealpha=0.9)

# ──────────────────────────────────────────────────────────────────────────────
# Sec 20: V10 Cluster-Scale Scalability
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec20():
    nodes = [1,2,4,8,16,32]
    SYSTEMS = list(COLORS.keys())
    weak_ttft = {
        "Cascade":       [12.3,38.3,37.2,55.4,53.7,49.9],
        "LMCache-Disk":  [46.9,213.4,214.2,214.1,214.2,214.9],
        "LMCache-Redis": [205.9,200.9,204.9,206.7,215.8,206.6],
        "PDC":           [49.6,213.5,217.7,211.4,214.4,216.6],
        "LLM-GPU":       [68.3,234.5,236.4,232.3,241.2,231.0],
        "HDF5-Indep":    [80.0,243.9,270.1,189.4,194.1,204.6],
    }
    weak_tput = {
        "Cascade":       [74.2,51.7,104.1,141.9,293.6,640.5],
        "LMCache-Disk":  [21.3,9.4,18.7,37.4,74.7,148.9],
        "LMCache-Redis": [4.9,10.0,19.5,38.7,74.2,154.9],
        "PDC":           [20.1,9.4,18.4,37.8,74.6,147.7],
        "LLM-GPU":       [14.6,8.5,16.9,34.4,66.3,138.5],
        "HDF5-Indep":    [12.5,8.2,14.8,42.2,82.4,156.4],
    }
    strong_ttft = {
        "Cascade":       [13.0,33.8,33.1,32.5,37.9,46.4],
        "LMCache-Disk":  [46.2,209.8,209.7,207.8,213.3,214.8],
        "LMCache-Redis": [209.8,203.3,205.9,207.3,206.8,207.3],
        "PDC":           [46.3,210.8,206.5,209.8,214.4,211.0],
        "LLM-GPU":       [126.7,230.9,226.6,232.4,238.6,231.2],
        "HDF5-Indep":    [77.0,275.9,260.6,271.8,240.9,188.3],
    }
    strong_tput = {
        "Cascade":       [75.9,59.1,120.6,244.2,419.8,685.2],
        "LMCache-Disk":  [21.7,9.5,19.1,38.5,75.0,148.9],
        "LMCache-Redis": [4.8,9.8,19.4,38.6,77.4,154.5],
        "PDC":           [21.6,9.5,19.4,38.1,74.6,151.6],
        "LLM-GPU":       [7.9,8.7,17.6,34.4,67.0,138.4],
        "HDF5-Indep":    [13.0,7.2,15.3,29.4,66.4,169.9],
    }
    fig, axes = plt.subplots(2,2, figsize=(28,20))
    for ax,data,ylabel in [
        (axes[0,0],weak_ttft,"TTFT (ms)"),
        (axes[0,1],weak_tput,"Throughput (req/s)"),
        (axes[1,0],strong_ttft,"TTFT (ms)"),
        (axes[1,1],strong_tput,"Throughput (req/s)"),
    ]:
        line_plot(ax, data, nodes)
        ax.set_xlabel("Nodes"); ax.set_ylabel(ylabel)
    axes[0,0].set_title("Weak Scaling — TTFT")
    axes[0,1].set_title("Weak Scaling — Throughput")
    axes[1,0].set_title("Strong Scaling — TTFT (128 req fixed)")
    axes[1,1].set_title("Strong Scaling — Throughput")
    legend_bottom(fig, SYSTEMS, ncol=3)
    fig.tight_layout(rect=[0,0.07,1,1])
    savefig(fig, "fig_sec20_v10_scalability.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 21: Qwen 320MB Weak+Strong Scaling
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec21():
    nodes = [1,2,4,8,16,32,64]
    SYSTEMS = list(COLORS.keys())
    weak_ttft = {
        "Cascade":       [33.68,74.83,84.09,100.82,82.69,80.91,86.55],
        "LMCache-Disk":  [92.43,407.19,413.33,412.82,None,None,None],
        "LMCache-Redis": [398.46,395.21,393.16,388.62,386.06,392.35,390.51],
        "PDC":           [89.76,412.61,411.75,414.66,412.81,412.27,412.18],
        "LLM-GPU":       [132.27,445.96,450.58,451.80,449.76,449.35,448.36],
        "HDF5-Indep":    [191.47,513.71,570.40,636.75,957.56,1523.00,3097.17],
    }
    weak_tput = {
        "Cascade":       [29.65,26.43,49.26,87.23,202.19,407.32,771.76],
        "LMCache-Disk":  [10.81,4.91,9.68,19.38,None,None,None],
        "LMCache-Redis": [2.51,5.07,10.19,20.60,41.48,81.64,163.99],
        "PDC":           [11.13,4.85,9.71,19.29,38.76,77.62,155.29],
        "LLM-GPU":       [7.56,4.48,8.88,17.71,35.58,71.21,142.75],
        "HDF5-Indep":    [5.22,3.89,7.15,13.09,18.88,26.98,30.42],
    }
    strong_ttft = {
        "Cascade":       [68.14,68.20,83.50,65.94,86.35,102.26,56.47],
        "LMCache-Disk":  [87.69,406.41,412.67,404.53,None,None,None],
        "LMCache-Redis": [369.30,388.95,393.39,388.51,387.34,394.07,389.85],
        "PDC":           [91.20,314.00,403.17,409.19,409.58,407.67,411.24],
        "LLM-GPU":       [305.94,475.77,446.47,449.41,453.54,450.97,452.62],
        "HDF5-Indep":    [189.03,506.99,592.23,707.35,1041.89,1524.88,2816.45],
    }
    strong_tput = {
        "Cascade":       [14.67,29.33,54.39,180.94,491.59,719.54,1072.15],
        "LMCache-Disk":  [11.40,4.92,9.69,19.78,None,None,None],
        "LMCache-Redis": [2.71,5.14,10.17,20.60,41.35,81.27,164.39],
        "PDC":           [10.96,6.37,9.92,19.55,39.06,78.49,155.61],
        "LLM-GPU":       [3.27,4.21,8.96,17.80,35.28,70.95,141.35],
        "HDF5-Indep":    [5.29,3.94,6.84,11.88,17.52,28.07,38.44],
    }
    fig, axes = plt.subplots(2,2, figsize=(30,22))
    for ax,data,ylabel in [
        (axes[0,0],weak_ttft,"TTFT (ms)"),
        (axes[0,1],weak_tput,"Throughput (req/s)"),
        (axes[1,0],strong_ttft,"TTFT (ms)"),
        (axes[1,1],strong_tput,"Throughput (req/s)"),
    ]:
        line_plot(ax, data, nodes)
        ax.set_xlabel("Nodes"); ax.set_ylabel(ylabel)
    axes[0,0].set_title("Weak Scaling — TTFT (8 req/node)")
    axes[0,1].set_title("Weak Scaling — Throughput (8 req/node)")
    axes[1,0].set_title("Strong Scaling — TTFT (128 req fixed)")
    axes[1,1].set_title("Strong Scaling — Throughput (128 req fixed)")
    legend_bottom(fig, SYSTEMS, ncol=3)
    fig.tight_layout(rect=[0,0.06,1,1])
    savefig(fig, "fig_sec21_qwen_scalability.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 22: Prefix Sharing Performance
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec22():
    nodes = [1,2,4,8,16,32,64]
    SYSTEMS = list(COLORS.keys())
    ttft = {
        "Cascade":       [11.3,21.4,30.2,34.1,53.2,105.8,184.2],
        "LMCache-Disk":  [45.8,126.4,164.9,187.2,197.4,206.2,226.8],
        "PDC":           [46.1,127.8,164.6,185.0,207.9,212.2,217.8],
        "LLM-GPU":       [75.3,147.5,185.3,204.0,223.5,221.6,241.0],
        "HDF5-Indep":    [100.4,262.0,267.2,271.2,234.3,256.6,322.9],
        "LMCache-Redis": [213.9,194.9,204.2,352.6,688.8,1360.9,2594.2],
    }
    bw = {
        "Cascade":       [14.1,20.8,24.4,45.7,55.0,61.2,73.1],
        "LMCache-Disk":  [3.5,4.2,5.7,8.8,14.9,27.1,44.4],
        "PDC":           [3.5,4.2,5.8,9.0,14.4,26.2,47.3],
        "LLM-GPU":       [2.1,3.0,4.4,7.2,12.5,24.3,43.6],
        "HDF5-Indep":    [1.6,1.2,2.4,4.7,11.0,20.1,32.0],
        "LMCache-Redis": [0.7,1.6,3.1,3.6,3.7,3.8,3.9],
    }
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(28,12))
    line_plot(ax1, ttft, nodes); ax1.set_xlabel("Nodes"); ax1.set_ylabel("TTFT (ms)"); ax1.set_title("Prefix Sharing — TTFT")
    line_plot(ax2, bw, nodes);   ax2.set_xlabel("Nodes"); ax2.set_ylabel("Agg. Bandwidth (GB/s)"); ax2.set_title("Prefix Sharing — Bandwidth")
    legend_bottom(fig, SYSTEMS, ncol=3)
    fig.tight_layout(rect=[0,0.08,1,1])
    savefig(fig, "fig_sec22_prefix_sharing.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 23: Hot/Warm/Cold Recovery
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec23():
    systems = ["Cascade","PDC","LMCache-Disk","LLM-GPU","HDF5-Indep","LMCache-Redis"]
    tiers = ["HOT","WARM","COLD"]
    lat = {
        "8N":  {"Cascade":[16.52,15.32,15.41],"PDC":[47.10,55.35,155.24],"LMCache-Disk":[48.20,56.81,144.93],"LLM-GPU":[77.35,77.01,76.75],"HDF5-Indep":[189.70,86.78,189.89],"LMCache-Redis":[239.28,406.93,213.51]},
        "16N": {"Cascade":[13.78,12.77,12.86],"PDC":[47.66,56.22,155.87],"LMCache-Disk":[46.90,55.33,55.11],"LLM-GPU":[77.01,77.02,77.26],"HDF5-Indep":[190.34,85.85,187.67],"LMCache-Redis":[898.37,740.42,209.49]},
        "32N": {"Cascade":[10.47,9.48,9.58],"PDC":[47.73,55.91,157.32],"LMCache-Disk":[46.88,54.67,134.84],"LLM-GPU":[77.16,77.01,62.80],"HDF5-Indep":[101.89,109.88,225.13],"LMCache-Redis":[None,None,None]},
    }
    fig, axes = plt.subplots(1,3, figsize=(32,13))
    for ax, (scale, data) in zip(axes, lat.items()):
        x = np.arange(len(tiers)); w = 0.12
        n = len(systems); offs = np.linspace(-(n-1)*w/2,(n-1)*w/2,n)
        for i,sys in enumerate(systems):
            vals = [v if v else 0 for v in data[sys]]
            ax.bar(x+offs[i], vals, width=w*0.88,
                   color=COLORS.get(sys,"#7F8C8D"), alpha=0.88)
        ax.set_xticks(x); ax.set_xticklabels(tiers)
        ax.set_ylabel("Latency (ms)"); ax.set_title(scale)
    legend_bottom(fig, systems, ncol=3)
    fig.tight_layout(rect=[0,0.08,1,1])
    savefig(fig, "fig_sec23_hot_warm_cold.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 24: Semantic Eviction Stability
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec24():
    nodes = [1,2,4,8]
    SYSTEMS = ["Cascade","LMCache-Disk","PDC","LLM-GPU","HDF5-Indep"]
    data = {
        "Cascade":      [98.44,12.81,14.40,13.22],
        "LMCache-Disk": [97.12,104.59,87.18,48.77],
        "PDC":          [126.60,92.75,134.37,121.77],
        "LLM-GPU":      [142.01,107.88,143.41,127.49],
        "HDF5-Indep":   [168.87,191.97,201.01,199.30],
    }
    fig, ax = plt.subplots(figsize=(18,12))
    line_plot(ax, data, nodes)
    ax.fill_between(range(len(nodes)), data["Cascade"], alpha=0.12, color=COLORS["Cascade"])
    ax.set_xlabel("Nodes"); ax.set_ylabel("TTFT of Protected Prefix (ms)")
    ax.set_title("1.5× Oversubscription — Protected Prefix TTFT (lower=better)")
    legend_bottom(fig, SYSTEMS, ncol=3)
    fig.tight_layout(rect=[0,0.08,1,1])
    savefig(fig, "fig_sec24_semantic_eviction.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 25: Tail Latency — Full (Avg, P50, P95, P99, P99.9)
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec25():
    nodes = [1,2,4,8,16,32]
    SYSTEMS = list(COLORS.keys())
    metrics = {
        "Avg (ms)": {
            "Cascade":       [12.78,30.54,27.48,30.22,35.18,33.23],
            "HDF5-Indep":    [103.29,81.36,54.57,55.00,107.07,108.73],
            "LMCache-Disk":  [48.26,92.12,118.82,140.73,153.67,163.63],
            "PDC":           [47.24,95.41,116.77,138.77,157.16,164.30],
            "LLM-GPU":       [69.52,101.11,162.92,190.52,236.29,239.35],
            "LMCache-Redis": [210.41,196.98,204.62,415.41,None,None],
        },
        "P50 (ms)": {
            "Cascade":       [12.75,28.49,30.71,31.42,32.74,32.09],
            "HDF5-Indep":    [90.63,13.88,1.61,2.92,6.13,13.11],
            "LMCache-Disk":  [46.33,53.64,152.39,154.19,154.82,156.59],
            "PDC":           [45.32,54.25,151.78,153.88,155.61,157.14],
            "LLM-GPU":       [57.85,86.22,172.79,176.08,226.95,235.75],
            "LMCache-Redis": [210.00,194.06,197.49,408.19,None,None],
        },
        "P99 (ms)": {
            "Cascade":       [13.00,83.77,48.21,48.22,76.13,48.81],
            "HDF5-Indep":    [146.48,249.49,251.96,247.86,253.62,248.07],
            "LMCache-Disk":  [60.34,214.55,218.27,218.69,218.52,218.59],
            "PDC":           [58.63,217.69,217.58,218.00,219.37,219.08],
            "LLM-GPU":       [86.99,239.72,471.06,466.25,996.44,603.04],
            "LMCache-Redis": [227.02,236.16,267.04,622.07,None,None],
        },
        "P99.9 (ms)": {
            "Cascade":       [17.44,88.64,50.14,50.95,87.32,52.79],
            "HDF5-Indep":    [156.44,3205.58,4914.96,13973.19,44111.95,43953.64],
            "LMCache-Disk":  [66.07,217.55,227.06,229.37,230.68,239.22],
            "PDC":           [59.61,227.12,241.15,228.97,282.67,252.02],
            "LLM-GPU":       [93.00,244.02,1095.30,994.90,1337.46,1122.73],
            "LMCache-Redis": [260.39,248.64,292.55,679.00,None,None],
        },
    }
    fig, axes = plt.subplots(2,2, figsize=(30,22))
    for ax, (mname, mdata) in zip(axes.flat, metrics.items()):
        line_plot(ax, mdata, nodes)
        if "P99" in mname: ax.set_yscale("log")
        ax.set_xlabel("Nodes"); ax.set_ylabel(mname)
        ax.set_title(mname + (" (log)" if "P99" in mname else ""))
    legend_bottom(fig, SYSTEMS, ncol=3)
    fig.tight_layout(rect=[0,0.07,1,1])
    savefig(fig, "fig_sec25_tail_latency.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 27: YCSB Mixed R/W
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec27():
    systems = ["Cascade","PDC","LMCache-Disk","LLM-GPU","HDF5-Indep","LMCache-Redis"]
    workloads = ["Workload A\n(95R/5W)","Workload B\n(50R/50W)","Workload C\n(Scan)"]
    ops_8N  = {"Cascade":[3288.7,376.0,53427.5],"PDC":[1010.5,289.2,11700.0],"LMCache-Disk":[1048.9,317.7,2179.0],"HDF5-Indep":[649.8,229.5,2249.9],"LMCache-Redis":[216.3,123.7,226.5],"LLM-GPU":[1222.6,309.8,11525.4]}
    ops_16N = {"Cascade":[6162.8,703.0,185042.3],"PDC":[1326.2,590.5,35827.4],"LMCache-Disk":[1416.6,588.4,4524.4],"HDF5-Indep":[463.3,305.0,733.4],"LMCache-Redis":[200.6,132.1,215.6],"LLM-GPU":[1223.9,627.5,25631.1]}
    p99_8N  = {"Cascade":[45.0,72.7,1.3],"PDC":[47.0,52.0,6.0],"LMCache-Disk":[40.0,50.6,20.5],"HDF5-Indep":[55.5,60.9,17.7],"LMCache-Redis":[83.9,115.1,58.4],"LLM-GPU":[41.9,54.9,2.9]}
    p99_16N = {"Cascade":[50.3,104.4,1.3],"PDC":[47.5,51.8,6.0],"LMCache-Disk":[39.9,46.8,19.3],"HDF5-Indep":[73.9,90.9,51.7],"LMCache-Redis":[136.0,208.0,118.7],"LLM-GPU":[47.2,55.0,2.9]}

    fig, axes = plt.subplots(2,2, figsize=(30,22))
    for ax, data, ylabel, title in [
        (axes[0,0], ops_8N,  "Ops/sec (log)", "8N — Throughput"),
        (axes[0,1], ops_16N, "Ops/sec (log)", "16N — Throughput"),
        (axes[1,0], p99_8N,  "P99 Latency (ms)", "8N — P99 Latency"),
        (axes[1,1], p99_16N, "P99 Latency (ms)", "16N — P99 Latency"),
    ]:
        x = np.arange(len(workloads)); w = 0.12; n = len(systems)
        offs = np.linspace(-(n-1)*w/2,(n-1)*w/2,n)
        for i,sys in enumerate(systems):
            ax.bar(x+offs[i], data[sys], width=w*0.88,
                   color=COLORS.get(sys,"#7F8C8D"), alpha=0.88)
        if "log" in ylabel: ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(workloads)
        ax.set_ylabel(ylabel); ax.set_title(title)
    legend_bottom(fig, systems, ncol=3)
    fig.tight_layout(rect=[0,0.07,1,1])
    savefig(fig, "fig_sec27_ycsb_mixed.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 28: Index Lookup Scalability
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec28():
    scales = ["1K","10K","100K","500K"]
    SYSTEMS = list(COLORS.keys())
    p99 = {
        "Cascade":       [0.05,0.02,0.04,0.04],
        "LMCache-Disk":  [4.30,7.54,4.52,8.18],
        "LLM-GPU":       [3.67,2.92,2.39,2.10],
        "PDC":           [2.41,1.83,1.86,2.04],
        "LMCache-Redis": [0.43,0.25,0.25,0.29],
        "HDF5-Indep":    [19.03,24.25,51.25,82.75],
    }
    fig, ax = plt.subplots(figsize=(18,12))
    xi = list(range(len(scales)))
    for sys, vals in p99.items():
        lw = LW_CAS if sys=="Cascade" else LW_BASE
        ms = MS_CAS if sys=="Cascade" else MS_BASE
        ax.plot(xi, vals, marker=MK.get(sys,"o"), color=COLORS.get(sys,"#7F8C8D"),
                linewidth=lw, markersize=ms)
    ax.set_xticks(xi); ax.set_xticklabels(scales)
    ax.set_xlabel("Number of Stored Blocks"); ax.set_ylabel("P99 Lookup Latency (ms)")
    ax.set_title("O(1) Index — P99 Latency vs Block Count")
    legend_bottom(fig, list(p99.keys()), ncol=3)
    fig.tight_layout(rect=[0,0.08,1,1])
    savefig(fig, "fig_sec28_index_lookup.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 29.1: Multi-Node Index Scalability (8N, 16MB blocks)
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec29_1():
    scales = ["1K\n(16GB)","10K\n(160GB)","50K\n(800GB)"]
    systems_29 = ["Cascade (HOT)","Cascade (COLD)","LMCache","PDC","LLM-GPU","HDF5-Indep","LMCache-Redis"]
    colors_29 = {
        "Cascade (HOT)":  COLORS["Cascade"],
        "Cascade (COLD)": "#C0392B",
        "LMCache":        COLORS["LMCache-Disk"],
        "PDC":            COLORS["PDC"],
        "LLM-GPU":        COLORS["LLM-GPU"],
        "HDF5-Indep":     COLORS["HDF5-Indep"],
        "LMCache-Redis":  COLORS["LMCache-Redis"],
    }
    mk29 = {"Cascade (HOT)":"o","Cascade (COLD)":"*","LMCache":"s","PDC":"^","LLM-GPU":"v","HDF5-Indep":"P","LMCache-Redis":"D"}
    p50 = {
        "Cascade (HOT)":  [0.00,0.00,0.01],
        "Cascade (COLD)": [0.00,0.01,0.01],
        "LMCache":        [23.67,22.58,22.27],
        "PDC":            [22.39,21.33,22.38],
        "LLM-GPU":        [26.76,25.17,23.58],
        "HDF5-Indep":     [2.83,3.18,3.25],
        "LMCache-Redis":  [24.04,22.75,19.64],
    }
    agg_bw = {
        "Cascade (HOT)":  [1899.91,2297.68,1798.70],
        "Cascade (COLD)": [2408.04,444.30,403.71],
        "LMCache":        [5.86,6.06,6.38],
        "PDC":            [6.26,6.56,6.66],
        "LLM-GPU":        [5.29,6.06,3.30],
        "HDF5-Indep":     [0.84,0.08,0.02],
        "LMCache-Redis":  [4.78,5.53,8.04],
    }
    xi = list(range(3))
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(28,13))
    for sys, vals in p50.items():
        lw = LW_CAS if "Cascade" in sys else LW_BASE
        ms = MS_CAS if "Cascade" in sys else MS_BASE
        ax1.plot(xi, vals, marker=mk29[sys], color=colors_29[sys], linewidth=lw, markersize=ms)
    ax1.set_xticks(xi); ax1.set_xticklabels(scales)
    ax1.set_xlabel("Scale"); ax1.set_ylabel("P50 TTFT (ms)"); ax1.set_title("P50 Latency — 8N, 16MB blocks")

    for sys, vals in agg_bw.items():
        lw = LW_CAS if "Cascade" in sys else LW_BASE
        ms = MS_CAS if "Cascade" in sys else MS_BASE
        ax2.plot(xi, vals, marker=mk29[sys], color=colors_29[sys], linewidth=lw, markersize=ms)
    ax2.set_yscale("log"); ax2.set_xticks(xi); ax2.set_xticklabels(scales)
    ax2.set_xlabel("Scale"); ax2.set_ylabel("Agg. Bandwidth (GB/s, log)"); ax2.set_title("Aggregate Bandwidth — 8N, 16MB blocks")

    handles = [mpatches.Patch(color=colors_29[s], label=s) for s in systems_29]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5,0), fontsize=FS_SM)
    fig.tight_layout(rect=[0,0.08,1,1])
    savefig(fig, "fig_sec29_1_multinode_index.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 29.2: Single-Node Disk Mode
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec29_2():
    systems = ["Cascade\n(Disk-COLD)","Cascade\n(Disk-HOT)","LMCache","PDC","HDF5-Indep","LMCache-Redis"]
    bw   = [6.76,7.03,0.94,0.95,0.93,8.21]
    p50  = [0.02,0.02,18.30,17.92,17.12,0.06]
    colors = [COLORS["Cascade"],COLORS["Cascade"],COLORS["LMCache-Disk"],COLORS["PDC"],COLORS["HDF5-Indep"],COLORS["LMCache-Redis"]]
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(26,13))
    x = np.arange(len(systems))
    b1 = ax1.bar(x, bw, color=colors, alpha=0.88, edgecolor="white", linewidth=1.5)
    ax1.set_xticks(x); ax1.set_xticklabels(systems, rotation=15, ha="right")
    ax1.set_ylabel("Aggregate Bandwidth (GB/s)"); ax1.set_title("Bandwidth (higher=better)")
    for bar,v in zip(b1,bw):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{v:.2f}", ha="center", fontsize=FS_ANN, fontweight="bold")
    b2 = ax2.bar(x, p50, color=colors, alpha=0.88, edgecolor="white", linewidth=1.5)
    ax2.set_xticks(x); ax2.set_xticklabels(systems, rotation=15, ha="right")
    ax2.set_ylabel("P50 TTFT (ms)"); ax2.set_title("P50 Latency (lower=better)")
    for bar,v in zip(b2,p50):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{v:.2f}", ha="center", fontsize=FS_ANN, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig_sec29_2_singlenode_disk.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sec 30: Time Breakdown
# ──────────────────────────────────────────────────────────────────────────────
def fig_sec30():
    fig, axes = plt.subplots(1,3, figsize=(34,12))

    # 30a: Per-tier BW — evenly spaced
    ax = axes[0]
    nodes4 = [1,2,4,8]
    xi4 = list(range(4)); xl4 = [f"{n}N" for n in nodes4]
    local_bw  = [11.27,16.23,12.46,16.23]
    remote_bw = [None, 5.41, 2.32, 5.39]
    rxi = [xi4[i] for i,v in enumerate(remote_bw) if v is not None]
    ry  = [v for v in remote_bw if v is not None]
    ax.plot(xi4, local_bw, "o-", color=COLORS["Cascade"], lw=LW_CAS, ms=MS_CAS, label="Local GPU")
    ax.plot(rxi, ry, "s--", color="#E67E22", lw=LW_BASE+0.5, ms=MS_BASE+2, label="Remote RDMA")
    ax.axhline(0.93, color="#7F8C8D", linestyle=":", lw=2, label="Lustre baseline (0.93 GB/s)")
    ax.set_xticks(xi4); ax.set_xticklabels(xl4)
    ax.set_xlabel("Nodes"); ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Per-tier Bandwidth")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5,-0.22), ncol=1, fontsize=FS_SM)

    # 30b: Phase composition bar
    ax = axes[1]
    phases = ["Index\nLookup","Data\nTransfer","Python\nDeser"]
    pcts   = [0.004, 99.992, 0.004]
    bcolors= ["#3498DB", COLORS["Cascade"], "#27AE60"]
    bars = ax.bar(phases, pcts, color=bcolors, alpha=0.88, width=0.5)
    for bar,p,lbl in zip(bars, pcts, ["~1 μs","~19,000 μs","~0.5 μs"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{p:.3f}%",
                ha="center", fontsize=FS_ANN, fontweight="bold")
        ax.text(bar.get_x()+bar.get_width()/2, max(bar.get_height()/2,5),
                lbl, ha="center", va="center", fontsize=FS_ANN, color="white", fontweight="bold")
    ax.set_ylabel("% of E2E Time"); ax.set_ylim(0,115)
    ax.set_title("Time Composition (Local Read)")

    # 30c: Realistic E2E stacked bar — evenly spaced 1N,2N,4N,8N
    ax = axes[2]
    nodes_e = [1,2,4,8]; xi_e = list(range(4))
    p_local  = [1.0, 0.5, 0.25, 0.125]
    p_remote = [0.0, 0.5, 0.75, 0.875]
    T_loc, T_rem = 19.3, 58.0
    local_ms  = [p*T_loc  for p in p_local]
    remote_ms = [p*T_rem  for p in p_remote]
    ax.bar(xi_e, local_ms,  label="Local GPU",  color=COLORS["Cascade"], alpha=0.88)
    ax.bar(xi_e, remote_ms, bottom=local_ms,    label="Remote RDMA",     color="#E67E22", alpha=0.88)
    total = [l+r for l,r in zip(local_ms,remote_ms)]
    for i,(t,rp) in enumerate(zip(total, p_remote)):
        ax.text(i, t+1.5, f"{t:.0f}ms", ha="center", fontsize=FS_ANN, fontweight="bold")
        if rp > 0:
            ax.text(i, local_ms[i]+remote_ms[i]/2, f"RDMA\n{rp*100:.0f}%",
                    ha="center", va="center", fontsize=FS_ANN-2, color="white")
    ax.set_xticks(xi_e); ax.set_xticklabels([f"{n}N" for n in nodes_e])
    ax.set_xlabel("Nodes"); ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("Realistic E2E Model\n(no Locality Promotion)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5,-0.22), ncol=1, fontsize=FS_SM)

    fig.tight_layout(pad=3.0)
    savefig(fig, "fig_sec30_time_breakdown.png")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Sections 20-30 figures...")
    fig_sec20()
    fig_sec21()
    fig_sec22()
    fig_sec23()
    fig_sec24()
    fig_sec25()
    fig_sec27()
    fig_sec28()
    fig_sec29_1()
    fig_sec29_2()
    fig_sec30()
    print(f"\n✅ All figures saved to {FIG_DIR}/")
