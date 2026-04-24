import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

colors = {
    'Cascade': '#E63946',
    'HDF5': '#1D3557',
    'LMCache-Disk': '#457B9D',
    'PDC': '#A8DADC',
    'LMCache-Redis': '#F4A261',
    'LLM-GPU': '#2A9D8F'
}

def plot_20_scalability():

    weak_data = {
        'Nodes': [1, 2, 4, 8, 16, 32, 64],
        'Cascade': [66.7, 71.7, 93.2, 195.0, 422.4, 823.8, 1660.0],
        'LMCache-Disk': [21.3, 9.4, 18.7, 37.4, 74.7, 148.9, 297.3],
        'LMCache-Redis': [4.9, 10.0, 19.5, 38.7, 74.2, 154.9, 309.0],
        'PDC': [20.1, 9.4, 18.4, 37.8, 74.6, 147.7, 300.6],
        'LLM-GPU': [14.6, 8.5, 16.9, 34.4, 66.3, 138.5, 276.4],
        'HDF5': [12.5, 8.2, 14.8, 42.2, 82.4, 156.4, 312.5]
    }

    strong_data = {
        'Nodes': [1, 2, 4, 8, 16, 32, 64],
        'Cascade': [10.0, 36.8, 34.5, 43.0, 46.4, 73.2, 88.9],
        'LMCache-Disk': [46.2, 209.8, 209.7, 207.8, 213.3, 214.8, 214.6],
        'LMCache-Redis': [209.8, 203.3, 205.9, 207.3, 206.8, 207.3, 209.3],
        'PDC': [46.3, 210.8, 206.5, 209.8, 214.4, 211.0, 211.4],
        'LLM-GPU': [126.7, 230.9, 226.6, 232.4, 238.6, 231.2, 227.8],
        'HDF5': [77.0, 275.9, 260.6, 271.8, 240.9, 188.3, 187.2]
    }

    weak_df = pd.DataFrame(weak_data)
    strong_df = pd.DataFrame(strong_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for col in weak_df.columns[1:]:
        ax1.plot(weak_df['Nodes'].astype(str), weak_df[col], marker='o', linewidth=2, color=colors[col], label=col)

    ax1.set_title('Cluster-Scale Weak Scaling (Throughput)', fontweight='bold')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Aggregate Throughput (req/s)')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    for col in strong_df.columns[1:]:
        ax2.plot(strong_df['Nodes'].astype(str), strong_df[col], marker='o', linewidth=2, color=colors[col], label=col)

    ax2.set_title('Cluster-Scale Strong Scaling (TTFT)', fontweight='bold')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Time-To-First-Token (ms)')
    ax2.set_ylim(0, 300)
    ax2.grid(True, ls="--", alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_20_scalability.png', bbox_inches='tight')
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_20_scalability.pdf', bbox_inches='tight')
    plt.close()

def plot_23_tiering():

    data = {
        'System': ['Cascade', 'HDF5', 'PDC', 'LLM-GPU', 'LMCache-Disk'],
        'HOT_Lat': [13.82, 151.98, 48.79, 130.54, 46.80],
        'WARM_Lat': [12.86, 160.14, 57.02, 130.23, 54.85],
        'COLD_Lat': [12.85, 275.47, 157.78, 131.36, 83.23]
    }
    df = pd.DataFrame(data)
    df = df.set_index('System')

    df = df.reindex(['Cascade', 'LMCache-Disk', 'PDC', 'LLM-GPU', 'HDF5'])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df.index))
    width = 0.25

    rects1 = ax.bar(x - width, df['HOT_Lat'], width, label='HOT (OS Cache/HBM)', color='#E63946', edgecolor='white')
    rects2 = ax.bar(x, df['WARM_Lat'], width, label='WARM (DRAM/RDMA)', color='#A8DADC', edgecolor='white')
    rects3 = ax.bar(x + width, df['COLD_Lat'], width, label='COLD (Disk/Lustre)', color='#1D3557', edgecolor='white')

    ax.set_ylabel('Latency (ms)')
    ax.set_title('Hierarchical Tiering Recovery Latency (64 Nodes)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    for i, rect in enumerate(rects3):
        height = rect.get_height()
        if i == 0 or i == 4:
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    cascade_cold = df.loc['Cascade', 'COLD_Lat']
    hdf5_cold = df.loc['HDF5', 'COLD_Lat']
    speedup = hdf5_cold / cascade_cold

    plt.tight_layout()
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_23_tiering.png', bbox_inches='tight')
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_23_tiering.pdf', bbox_inches='tight')
    plt.close()

def plot_24_oversubscription():
    data = {
        'Nodes': [1, 2, 4, 8, 16, 32, 64],
        'Cascade': [98.44, 12.81, 14.40, 13.22, 12.89, 17.74, 22.95],
        'LMCache-Disk': [97.12, 104.59, 87.18, 48.77, 46.84, 47.35, 46.92],
        'PDC': [126.60, 92.75, 134.37, 121.77, 47.71, 152.10, 122.60],
        'LLM-GPU': [142.01, 107.88, 143.41, 127.49, 147.88, 152.11, 197.42],
        'HDF5': [168.87, 191.97, 201.01, 199.30, 161.94, 181.01, 183.20]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))

    for col in df.columns[1:]:
        ax.plot(df['Nodes'].astype(str), df[col], marker='o', linewidth=2, color=colors[col], label=col)

    ax.set_title('Semantic Eviction Stability (1.5x Cluster Oversubscription)', fontweight='bold')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('TTFT for "Protected Prefix"(ms)')
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_24_oversubscription.png', bbox_inches='tight')
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_24_oversubscription.pdf', bbox_inches='tight')
    plt.close()

def plot_27_mixed_workload():
    data = {
        'System': ['Cascade', 'LLM-GPU', 'LMCache-Disk', 'PDC', 'HDF5', 'Redis'],
        'Workload A (95/5)': [3288.7, 1222.6, 1048.9, 1010.5, 649.8, 216.3],
        'Workload B (50/50)': [376.0, 309.8, 317.7, 289.2, 229.5, 123.7],
        'Workload C (Scan)': [53427.5, 11525.4, 2179.0, 11700.0, 2249.9, 226.5]
    }
    df = pd.DataFrame(data)
    df = df.set_index('System')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(df.index))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df['Workload A (95/5)'], width, label='Workload A (95% Read)', color='#457B9D')
    rects2 = ax1.bar(x + width/2, df['Workload B (50/50)'], width, label='Workload B (50% Read)', color='#F4A261')

    ax1.set_ylabel('Ops / sec')
    ax1.set_title('Write-Heavy & Read-Heavy Mix (8 Nodes)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index, rotation=45, ha='right')
    ax1.legend(frameon=True, fancybox=True, shadow=True)

    color_map = [colors.get(sys, '#cccccc') for sys in df.index]
    rects3 = ax2.bar(x, df['Workload C (Scan)'], color=color_map)
    ax2.set_ylabel('Ops / sec (Log Scale)')
    ax2.set_title('Sequential Scan Mix (8 Nodes)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df.index, rotation=45, ha='right')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_27_mixed_workload.png', bbox_inches='tight')
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_27_mixed_workload.pdf', bbox_inches='tight')
    plt.close()

def plot_30_9_deepcam():
    data = {
        'Nodes': [1, 2, 4, 8, 16, 32, 64],
        'HDF5': [227.9, 127.9, 71.6, 43.8, 28.6, 33.1, 13.7],
        'Cascade (Streaming)': [276.7, 158.8, 77.5, 52.7, 37.9, 23.1, 15.7]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df['Nodes'].astype(str), df['HDF5'], marker='s', linewidth=2, color=colors['HDF5'], label='HDF5-Indep (Baseline)')
    ax.plot(df['Nodes'].astype(str), df['Cascade (Streaming)'], marker='o', linewidth=2, color=colors['Cascade'], label='Cascade (Streaming Mode)')

    ax.set_title('DeepCAM Inference/Training Replica (512GB)', fontweight='bold')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Epoch Time (Seconds)')
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    ax.text(6, 40, "Both systems converge\n~14-15s at 64 Nodes", ha='right', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_30_deepcam.png', bbox_inches='tight')
    plt.savefig('${REPO_ROOT}/paper/Figures/eval_30_deepcam.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Evaluation Figures...")
    plot_20_scalability()
    print("1. Scalability Plotted")
    plot_23_tiering()
    print("2. Tiering Plotted")
    plot_24_oversubscription()
    print("3. Oversubscription Plotted")
    plot_27_mixed_workload()
    print("4. Mixed Workload Plotted")
    plot_30_9_deepcam()
    print("5. DeepCAM Plotted")
    print("All Evaluation Figures generated successfully.")
