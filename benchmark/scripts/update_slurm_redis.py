import glob
import os

slurm_files = glob.glob('benchmark/scripts/v6_amrex_n*.slurm') + glob.glob('benchmark/scripts/v6_lammps_n*.slurm')
redis_bin = "/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server"
redis_cli = "/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-cli"

for fpath in slurm_files:
    with open(fpath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith('source setup_env.sh'):
            new_lines.append(line)
            new_lines.append("\n# Start Redis on each node\n")
            new_lines.append(f"srun -n $SLURM_NNODES --ntasks-per-node=1 {redis_bin} --port 16379 --daemonize yes --maxmemory 100gb --maxmemory-policy allkeys-lru\n")
            new_lines.append("sleep 2\n")
        elif '--systems' in line:
            new_lines.append(line.replace('Cascade,HDF5,vLLM-GPU,PDC,LMCache', 'Cascade,HDF5,vLLM-GPU,PDC,LMCache,Redis'))
        else:
            new_lines.append(line)
    
    # Add cleanup at the end
    new_lines.append("\n# Cleanup Redis\n")
    new_lines.append(f"srun -n $SLURM_NNODES --ntasks-per-node=1 {redis_cli} -p 16379 shutdown nosave\n")

    with open(fpath, 'w') as f:
        f.writelines(new_lines)

print(f"Updated {len(slurm_files)} SLURM scripts.")
