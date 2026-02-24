#!/bin/bash
# submit_real_hpc_apps.sh
# Generates and submits SLURM jobs for LAMMPS and AMReX Proxy Apps

set -e

BASE_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
SCRIPT_DIR="$BASE_DIR/benchmark/scripts"
LOG_DIR="$BASE_DIR/benchmark/logs"
TIME_LIMIT="1:00:00"

mkdir -p "$LOG_DIR"

# LAMMPS Parameters (100M atoms per rank = ~8.4 GB per rank)
LAMMPS_ATOMS=100000000
LAMMPS_STEPS=5

# AMReX Parameters (Grid: 256^3, 5 vars = 640MB per block. 15 blocks = 9.6 GB per rank)
AMREX_GRID=256
AMREX_BLOCKS=15
AMREX_STEPS=5

for NODES in 8 4 2 1; do
    RANKS=$((NODES * 1)) # 1 rank per node for memory safety
    
    # -------------------------------------------------------------
    # 1. LAMMPS I/O Slurm Script
    # -------------------------------------------------------------
    LAMMPS_SLURM="$SCRIPT_DIR/v6_lammps_n${NODES}.slurm"
    cat << EOF > "$LAMMPS_SLURM"
#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t $TIME_LIMIT
#SBATCH -N $NODES
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J v6_lammps_n${NODES}
#SBATCH -D /pscratch/sd/s/sgkim/kcj/Cascade-kcj
#SBATCH -o benchmark/logs/lammps_n${NODES}_%j.out
#SBATCH -e benchmark/logs/lammps_n${NODES}_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

srun -n $RANKS --ntasks-per-node=1 -c 32 --gpus-per-node=4 \\
    python benchmark/scripts/v6_app_lammps_io.py \\
    --atoms_per_rank $LAMMPS_ATOMS \\
    --timesteps $LAMMPS_STEPS \\
    --systems Cascade,HDF5
EOF

    # -------------------------------------------------------------
    # 2. AMReX I/O Slurm Script
    # -------------------------------------------------------------
    AMREX_SLURM="$SCRIPT_DIR/v6_amrex_n${NODES}.slurm"
    cat << EOF > "$AMREX_SLURM"
#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t $TIME_LIMIT
#SBATCH -N $NODES
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J v6_amrex_n${NODES}
#SBATCH -D /pscratch/sd/s/sgkim/kcj/Cascade-kcj
#SBATCH -o benchmark/logs/amrex_n${NODES}_%j.out
#SBATCH -e benchmark/logs/amrex_n${NODES}_%j.err

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

srun -n $RANKS --ntasks-per-node=1 -c 32 --gpus-per-node=4 \\
    python benchmark/scripts/v6_app_amrex_io.py \\
    --grid_dim $AMREX_GRID \\
    --n_blocks $AMREX_BLOCKS \\
    --timesteps $AMREX_STEPS \\
    --systems Cascade,HDF5
EOF

    echo "Submitting LAMMPS and AMReX jobs for $NODES Nodes..."
    sbatch "$LAMMPS_SLURM"
    sbatch "$AMREX_SLURM"
done
