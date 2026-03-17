#!/bin/bash
# benchmark/scripts/v20_submit_debug_scaling.sh
# Creates and submits 6 separate jobs for 1,2,4,8 node scaling with variable blocks

SYSTEMS=("cascade" "lmcache" "pdc" "vllm-gpu" "hdf5" "redis")

for sys in "${SYSTEMS[@]}"; do
    cat <<EOF > benchmark/scripts/v20_var_debug_${sys}.slurm
#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -J v20_${sys}_dbg
#SBATCH -o benchmark/logs/v20_var_debug_${sys}_%j.out
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh

echo "=== 🏁 Variable Block Scaling: ${sys} (1,2,4,8 Nodes) ==="

NODES=(1 2 4 8)
for n in "\${NODES[@]}"; do
    echo "--------------------------------------------------------"
    echo "▶ Node Count: \$n"
    echo "--------------------------------------------------------"
    
    if [ "${sys}" == "redis" ]; then
        # Handle Redis Server
        mkdir -p benchmark/tmp/hosts_\$SLURM_JOB_ID
        hostname > benchmark/tmp/hosts_\$SLURM_JOB_ID/redis_host
        /global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/bin/redis-server --port 16379 --protected-mode no --save "" --appendonly no --maxmemory 100gb --maxmemory-policy allkeys-lru &
        REDIS_PID=\$!
        sleep 10
    fi

    srun -N \$n -n \$n python3 benchmark/scripts/v20_variable_block_bench.py \\
        --system ${sys} \\
        --block-size-mb 160 --num-write-blocks 50 --num-read-ops 200 --sigma 0.8 \\
        --redis-port 16379

    if [ "${sys}" == "redis" ]; then
        kill \$REDIS_PID 2>/dev/null || true
        rm -rf benchmark/tmp/hosts_\$SLURM_JOB_ID
        sleep 5
    fi
    
    sleep 5
done

EOF
    sbatch benchmark/scripts/v20_var_debug_${sys}.slurm
done

echo "✅ 6 debug scaling jobs (Cascade, LMCache, PDC, vLLM-GPU, HDF5, Redis) submitted."
