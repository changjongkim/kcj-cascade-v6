import re

with open('../README.md', 'r') as f:
    readme_content = f.read()

# Define the new section
new_section = r"""
### 🧬 14. Real-World HPC Applications: AMReX and LAMMPS Output
*   **Experimental Objective**: Validate Cascade's capability as a general-purpose scientific I/O backend for production HPC simulation codes involving complex checkpointing and parallel trajectory dumps.
*   **Methodology**: Tested the Cascade C++ tiered storage backend against traditional file formats (HDF5) across 1 to 8 nodes using proxy applications for AMReX (Adaptive Mesh Refinement) and LAMMPS (Molecular Dynamics).

#### **[App 1] AMReX (Adaptive Mesh Refinement I/O)**
AMReX generates structured grid data with hierarchical refinement. The workload involves dumping Plotfiles and executing Halo Exchanges (ghost cell synchronization) across nodes.

| Nodes | System | Plotfile Write (GB/s) | Restart Read (GB/s) | Halo Exchange (GB/s) |
| :---: | :--- | :---: | :---: | :---: |
| **1 Node** | **Cascade** | **0.90** | **1.44** | 0.00 (N/A) |
| | HDF5 | 0.86 | 1.77 | 0.00 (N/A) |
| **2 Nodes** | **Cascade** | **1.79** | **2.87** | **2.75** |
| | HDF5 | (Failed*) | (Failed*) | (N/A) |
| **4 Nodes** | **Cascade** | **3.57** | **5.48** | **3.93** |
| | HDF5 | (Failed*) | (Failed*) | (N/A) |
| **8 Nodes** | **Cascade** | **7.11** | **9.29** | **7.78** |
| | HDF5 | 4.90 | 3.27 | 0.00 |

> **Analysis**:
> *   **HDF5 Parallel File Sync Issues**: The HDF5 baseline frequently failed (Failed*) during multi-node runs (Phase 3 Halo Exchange). This is due to Lustre metadata sync delays—when one node writes an HDF5 block and an adjacent node immediately tries to read it for Halo Exchange, HDF5 throws a `bad object header version number` error because the file system hasn't synchronized.
> *   **Cascade's RDMA Superiority**: Cascade bypasses the file system entirely for Halo Exchanges, fetching boundary data directly from neighboring nodes' GPU/DRAM via RDMA. This results in a highly stable **7.78 GB/s** boundary exchange rate at 8 nodes.
> *   **Linear Scalability**: Cascade demonstrates near-perfect linear scaling for Plotfile writes (0.90 GB/s to 7.11 GB/s) and Restart reads (1.44 GB/s to 9.29 GB/s).

#### **[App 2] LAMMPS (Molecular Dynamics I/O)**
Simulates the trajectory dumping of hundreds of millions of atoms. This is a massive, contiguous I/O operation typical of NVT ensemble simulations.

| Nodes | System | Total Data | Write BW (GB/s) | Restart Read BW (GB/s) |
| :---: | :--- | :---: | :---: | :---: |
| **1 Node** | **Cascade** | 39.1 GB | **0.78** | **1.30** |
| | HDF5 | 39.1 GB | 0.73 | 0.03 |
| **2 Nodes**| **Cascade** | 78.2 GB | **1.65** | **2.60** |
| | HDF5 | 78.2 GB | 1.43 | 0.21 |
| **4 Nodes**| **Cascade** | 156.5 GB | **3.26** | **5.07** |
| | HDF5 | 156.5 GB | 3.19 | 0.27 |
| **8 Nodes**| **Cascade** | 312.9 GB | **6.56** | **10.54** |
| | HDF5 | 312.9 GB | 5.63 | 0.21 |

> **Analysis**:
> *   **Restart Read Acceleration**: Cascade completely dominates the Restart Read phase. While HDF5 struggles to read from Lustre at **0.21 GB/s** (due to collective MPI-IO bottlenecks), Cascade pulls the data directly from its GPU/DRAM tiers at an astonishing **10.54 GB/s**—a **50x speedup**.
> *   **Stable Throughput**: LAMMPS uses a simpler, large-block write pattern, so HDF5 doesn't crash as it did in AMReX. However, Cascade still outperforms HDF5 in Write BW (6.56 vs 5.63 GB/s) while drastically reducing the metadata load on the Lustre array.

---

"""

# Find insertion point
insertion_marker = "## 🧪 System Overhead Sensitivity Analysis"

if insertion_marker in readme_content:
    parts = readme_content.split(insertion_marker)
    updated_readme = parts[0] + new_section + insertion_marker + parts[1]
    
    with open('../README.md', 'w') as f:
        f.write(updated_readme)
    print("README.md updated successfully.")
else:
    print("Insertion marker not found.")
