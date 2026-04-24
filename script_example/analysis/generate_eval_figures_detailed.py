import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FixedLocator

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300

OUT_DIR = '${REPO_ROOT}/paper/Figures/eval_detailed'
os.makedirs(OUT_DIR, exist_ok=True)

colors = {
    'Cascade': '#E63946',
    'HDF5': '#1D3557',
    'LMCache-Disk': '#457B9D',
    'PDC': '#A8DADC',
    'LMCache-Redis': '#F4A261',
    'LLM-GPU': '#2A9D8F'
}

markers = {
    'Cascade': 'o',
    'HDF5': 's',
    'LMCache-Disk': '^',
    'PDC': 'v',
    'LMCache-Redis': 'D',
    'LLM-GPU': 'p'
}

def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)

def add_legend(ax, ncol=3):
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=ncol, frameon=True, fancybox=True, shadow=True)

def plot_scalability_160mb():
    nodes = [1, 2, 4, 8, 16, 32, 64]

    data_weak_ttft = {
        'Cascade': [13.9, 38.1, 47.0, 43.3, 37.8, 38.0, 39.3],
        'LMCache-Disk': [46.9, 213.4, 214.2, 214.1, 214.2, 214.9, 215.2],
        'LMCache-Redis': [205.9, 200.9, 204.9, 206.7, 215.8, 206.6, 207.2],
        'PDC': [49.6, 213.5, 217.7, 211.4, 214.4, 216.6, 212.9],
        'LLM-GPU': [68.3, 234.5, 236.4, 232.3, 241.2, 231.0, 231.5],
        'HDF5': [80.0, 243.9, 270.1, 189.4, 194.1, 204.6, 204.8]
    }

    data_weak_tp = {
        'Cascade': [66.7, 71.7, 93.2, 195.0, 422.4, 823.8, 1660.0],
        'LMCache-Disk': [21.3, 9.4, 18.7, 37.4, 74.7, 148.9, 297.3],
        'LMCache-Redis': [4.9, 10.0, 19.5, 38.7, 74.2, 154.9, 309.0],
        'PDC': [20.1, 9.4, 18.4, 37.8, 74.6, 147.7, 300.6],
        'LLM-GPU': [14.6, 8.5, 16.9, 34.4, 66.3, 138.5, 276.4],
        'HDF5': [12.5, 8.2, 14.8, 42.2, 82.4, 156.4, 312.5]
    }

    data_strong_ttft = {
        'Cascade': [10.0, 36.8, 44.5, 43.0, 46.4, 43.2, 42.9],
        'LMCache-Disk': [46.2, 209.8, 209.7, 207.8, 213.3, 214.8, 214.6],
        'LMCache-Redis': [209.8, 203.3, 205.9, 207.3, 206.8, 207.3, 209.3],
        'PDC': [46.3, 210.8, 206.5, 209.8, 214.4, 211.0, 211.4],
        'LLM-GPU': [126.7, 230.9, 226.6, 232.4, 238.6, 231.2, 227.8],
        'HDF5': [77.0, 275.9, 260.6, 271.8, 240.9, 188.3, 187.2]
    }

    data_strong_tp = {
        'Cascade': [99.6, 74.6, 116.0, 194.5, 340.3, 477.5, 745.1],
        'LMCache-Disk': [21.7, 9.5, 19.1, 38.5, 75.0, 148.9, 298.1],
        'LMCache-Redis': [4.8, 9.8, 19.4, 38.6, 77.4, 154.5, 306.1],
        'PDC': [21.6, 9.5, 19.4, 38.1, 74.6, 151.6, 302.5],
        'LLM-GPU': [7.9, 8.7, 17.6, 34.4, 67.0, 138.4, 280.8],
        'HDF5': [13.0, 7.2, 15.3, 29.4, 66.4, 169.9, 341.8]
    }

    datasets = [
        (data_weak_ttft, 'TTFT (ms) [Log Scale]', 'weak_ttft_160mb', True),
        (data_weak_tp, 'Aggregate Throughput (req/s)', 'weak_tp_160mb', True),
        (data_strong_ttft, 'TTFT (ms) [Log Scale]', 'strong_ttft_160mb', True),
        (data_strong_tp, 'Aggregate Throughput (req/s)', 'strong_tp_160mb', True),
    ]

    for data, ylabel, name, use_log in datasets:
        fig, ax = plt.subplots(figsize=(6, 4))
        for sys_name, vals in data.items():
            ax.plot(nodes, vals, marker=markers[sys_name], linewidth=2.5, color=colors[sys_name], label=sys_name, markersize=8)
        ax.set_xscale('log', base=2)
        if use_log:
            ax.set_yscale('log')
        ax.set_xticks(nodes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        add_legend(ax)
        save_fig(fig, f'scalability_{name}')

def plot_scalability_320mb():
    nodes = [1, 2, 4, 8, 16, 32, 64]

    data_weak_ttft = {
        'Cascade': [20.9, 108.7, 86.2, 98.8, 97.2, 92.9, 101.7],
        'LMCache-Disk': [92.4, 407.2, 413.3, 412.8, 414.0, 413.2, 408.1],
        'LMCache-Redis': [398.5, 395.2, 393.2, 388.6, 386.1, 392.4, 390.5],
        'PDC': [89.8, 412.6, 411.8, 414.7, 412.8, 412.3, 412.2],
        'LLM-GPU': [132.3, 446.0, 450.6, 451.8, 449.8, 449.4, 448.4],
        'HDF5': [191.5, 513.7, 570.4, 636.8, 957.6, 1523.0, 3097.2]
    }
    data_weak_tp = {
        'Cascade': [47.7, 19.5, 48.6, 89.5, 171.0, 356.6, 776.1],
        'LMCache-Disk': [10.8, 4.9, 9.7, 19.4, 38.7, 77.4, 156.9],
        'LMCache-Redis': [2.5, 5.1, 10.2, 20.6, 41.5, 81.6, 164.0],
        'PDC': [11.1, 4.9, 9.7, 19.3, 38.8, 77.6, 155.3],
        'LLM-GPU': [7.6, 4.5, 8.9, 17.7, 35.6, 71.2, 142.8],
        'HDF5': [5.2, 3.9, 7.2, 13.1, 18.9, 27.0, 30.4]
    }
    data_strong_ttft = {
        'Cascade': [26.1, 101.6, 87.6, 74.0, 93.2, 104.6, 115.8],
        'LMCache-Disk': [87.7, 406.4, 412.7, 404.5, 406.3, 410.6, 410.0],
        'LMCache-Redis': [369.3, 389.0, 393.4, 388.5, 387.3, 394.1, 389.9],
        'PDC': [91.2, 314.0, 403.2, 409.2, 409.6, 407.7, 411.2],
        'LLM-GPU': [305.9, 475.8, 446.5, 449.4, 453.5, 451.0, 452.6],
        'HDF5': [189.0, 507.0, 592.2, 707.4, 1041.9, 1524.9, 2816.5]
    }
    data_strong_tp = {
        'Cascade': [38.3, 17.1, 51.9, 108.1, 142.6, 297.5, 464.3],
        'LMCache-Disk': [11.4, 4.9, 9.7, 19.8, 39.4, 77.9, 156.1],
        'LMCache-Redis': [2.7, 5.1, 10.2, 20.6, 41.4, 81.3, 164.4],
        'PDC': [11.0, 6.4, 9.9, 19.6, 39.1, 78.5, 155.6],
        'LLM-GPU': [3.3, 4.2, 9.0, 17.8, 35.3, 71.0, 141.4],
        'HDF5': [5.3, 3.9, 6.8, 11.9, 17.5, 28.1, 38.4]
    }

    datasets = [
        (data_weak_ttft, 'TTFT (ms) [Log Scale]', 'weak_ttft_320mb', True),
        (data_weak_tp, 'Aggregate Throughput (req/s)', 'weak_tp_320mb', True),
        (data_strong_ttft, 'TTFT (ms) [Log Scale]', 'strong_ttft_320mb', True),
        (data_strong_tp, 'Aggregate Throughput (req/s)', 'strong_tp_320mb', True),
    ]

    for data, ylabel, name, use_log in datasets:
        fig, ax = plt.subplots(figsize=(6, 4))
        for sys_name, vals in data.items():
            ax.plot(nodes, vals, marker=markers[sys_name], linewidth=2.5, color=colors[sys_name], label=sys_name, markersize=8)
        ax.set_xscale('log', base=2)
        if use_log:
            ax.set_yscale('log')
        ax.set_xticks(nodes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        add_legend(ax)
        save_fig(fig, f'scalability_{name}')

def plot_prefix_sharing():
    nodes = [1, 2, 4, 8, 16, 32, 64]
    data_ttft = {
        'Cascade': [13.0, 13.6, 13.8, 13.9, 13.6, 13.3, 12.9],
        'LMCache-Disk': [45.8, 126.4, 164.9, 187.2, 197.4, 206.2, 226.8],
        'PDC': [46.1, 127.8, 164.6, 185.0, 207.9, 212.2, 217.8],
        'LLM-GPU': [75.3, 147.5, 185.3, 204.0, 223.5, 221.6, 241.0],
        'HDF5': [100.4, 262.0, 267.2, 271.2, 234.3, 256.6, 322.9],
        'LMCache-Redis': [213.9, 194.9, 204.2, 352.6, 688.8, 1360.9, 2594.2]
    }

    data_bw = {
        'Cascade': [12.3, 23.4, 46.8, 96.4, 209.4, 397.7, 824.4],
        'LMCache-Disk': [3.5, 4.2, 5.7, 8.8, 14.9, 27.1, 44.4],
        'PDC': [3.5, 4.2, 5.8, 9.0, 14.4, 26.2, 47.3],
        'LLM-GPU': [2.1, 3.0, 4.4, 7.2, 12.5, 24.3, 43.6],
        'HDF5': [1.6, 1.2, 2.4, 4.7, 11.0, 20.1, 32.0],
        'LMCache-Redis': [0.7, 1.6, 3.1, 3.6, 3.7, 3.8, 3.9]
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for sys_name, vals in data_ttft.items():
        ax.plot(nodes, vals, marker=markers[sys_name], linewidth=2.5, color=colors[sys_name], label=sys_name, markersize=8)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(nodes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Number of Nodes', fontweight='bold')
    ax.set_ylabel('TTFT (ms) [Log Scale]', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    add_legend(ax)
    save_fig(fig, 'prefix_sharing_ttft')

    fig, ax = plt.subplots(figsize=(6, 4))
    for sys_name, vals in data_bw.items():
        ax.plot(nodes, vals, marker=markers[sys_name], linewidth=2.5, color=colors[sys_name], label=sys_name, markersize=8)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(nodes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Number of Nodes', fontweight='bold')
    ax.set_ylabel('Aggregate Bandwidth (GB/s)', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    add_legend(ax)
    save_fig(fig, 'prefix_sharing_bw')

def plot_tiering_recovery():
    n_scales = ['8N', '16N', '32N', '64N']
    sys_list = ['Cascade', 'LMCache-Disk', 'PDC', 'LLM-GPU', 'HDF5']

    records = {
        '8N':  {'HOT': [16.5, 48.2, 47.1, 77.4, 189.7], 'WARM': [15.3, 56.8, 55.4, 77.0, 86.8], 'COLD': [15.4, 144.9, 155.2, 76.8, 189.9]},
        '16N': {'HOT': [13.8, 46.9, 47.7, 77.0, 190.3], 'WARM': [12.8, 55.3, 56.2, 77.0, 85.9], 'COLD': [12.9, 55.1, 155.9, 77.3, 187.7]},
        '32N': {'HOT': [10.5, 46.9, 47.7, 77.2, 101.9], 'WARM': [9.5,  54.7, 55.9, 77.0, 109.9], 'COLD': [9.6,  134.8, 157.3, 62.8, 225.1]},
        '64N': {'HOT': [13.8, 46.8, 48.8, 130.5, 152.0], 'WARM': [12.9, 54.9, 57.0, 130.2, 160.1], 'COLD': [12.9, 83.2, 157.8, 131.4, 275.5]}
    }

    width = 0.25
    x = np.arange(len(sys_list))

    for scale in n_scales:
        fig, ax = plt.subplots(figsize=(6, 4))

        hot = records[scale]['HOT']
        warm = records[scale]['WARM']
        cold = records[scale]['COLD']

        ax.bar(x - width, hot, width, label='HOT (OS Cache/HBM)', color='#E63946', edgecolor='white')
        ax.bar(x, warm, width, label='WARM (DRAM/RDMA)', color='#F4A261', edgecolor='white')
        ax.bar(x + width, cold, width, label='COLD (Disk/Lustre)', color='#1D3557', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(sys_list, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        casc_cold = cold[0]
        hdf5_cold = cold[4]
        speedup = hdf5_cold / casc_cold
        ax.annotate(f"{casc_cold}ms\n({speedup:.1f}x)", xy=(x[0]+width, casc_cold + max(cold)*0.05),
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

        add_legend(ax, ncol=3)
        save_fig(fig, f'tiering_latency_{scale}')

def plot_tail_latency():
    system = ['Cascade', 'LMCache-Disk', 'PDC', 'LLM-GPU', 'HDF5']
    p50 = [31.4, 154.2, 153.9, 176.1, 2.9]
    p99 = [48.2, 218.7, 218.0, 466.3, 247.9]
    p999 = [51.0, 229.4, 229.0, 994.9, 13973.2]

    x = np.arange(len(system))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width, p50, width, label='P50 (Median)', color='#A8DADC')
    ax.bar(x, p99, width, label='P99', color='#457B9D')
    ax.bar(x + width, p999, width, label='P99.9 (Tail)', color='#E63946')

    ax.set_xticks(x)
    ax.set_xticklabels(system, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylabel('Latency (ms) [Log Scale]', fontweight='bold')
    ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.3)

    ax.annotate("13.9 seconds!", xy=(x[-1]+width, p999[-1]), xytext=(x[-1]+width, p999[-1]/10),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center', va='top', fontweight='bold', color='#E63946')

    add_legend(ax, ncol=3)
    save_fig(fig, 'tail_latency_distribution')

def plot_index_overhead():
    blocks_str = ['1K', '10K', '100K', '500K']
    data = {
        'Cascade': [0.05, 0.02, 0.04, 0.04],
        'LMCache-Disk': [4.30, 7.54, 4.52, 8.18],
        'PDC': [2.41, 1.83, 1.86, 2.04],
        'LLM-GPU': [3.67, 2.92, 2.39, 2.10],
        'HDF5': [19.03, 24.25, 51.25, 82.75]
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for sys_name, vals in data.items():
        ax.plot(blocks_str, vals, marker=markers[sys_name], linewidth=2.5, color=colors[sys_name], label=sys_name, markersize=8)

    ax.set_ylabel('P99 Latency (ms)', fontweight='bold')
    ax.set_xlabel('Number of Unique KV Blocks', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    ax.annotate('$O(1)$ Hash Index (~0.04ms)', xy=(2, 0.04), xytext=(1.5, 0.01),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                ha='center', va='top', fontweight='bold')

    add_legend(ax, ncol=3)
    save_fig(fig, 'index_scalability_lookup')

if __name__ == '__main__':
    print("Generating comprehensive evaluation figures...")
    plot_scalability_160mb()
    plot_scalability_320mb()
    plot_prefix_sharing()
    plot_tiering_recovery()
    plot_tail_latency()
    plot_index_overhead()
    print("Completed generating all detailed figures.")
