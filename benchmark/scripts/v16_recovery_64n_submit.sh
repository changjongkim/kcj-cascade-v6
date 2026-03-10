#!/bin/bash
# Description: Submit 64n Recovery (Hot/Warm/Cold) Benchmarks for all systems
# Also submits 32n for LMCache-Redis as requested

SYSTEMS=("cascade" "hdf5" "pdc" "llm-gpu" "lmcache" "lmcache-redis")
TIME="00:40:00"
QUEUE="regular"
BLOCK_MB=160
REPO_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj"

cd $REPO_DIR

for SYS in "${SYSTEMS[@]}"; do
    for NODES in 64; do
        JOB_NAME="rec_${SYS}_${NODES}n"
        echo "Generating and submitting ${JOB_NAME}..."
        
        # Create a temp file
        TEMP_SLURM="benchmark/scripts/tmp_${JOB_NAME}.slurm"
        
        cat <<EOF > $TEMP_SLURM
#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q $QUEUE
#SBATCH -t $TIME
#SBATCH -N $NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J $JOB_NAME
#SBATCH -o benchmark/logs/${JOB_NAME}_%j.out
#SBATCH -e benchmark/logs/${JOB_NAME}_%j.err

cd $REPO_DIR
source setup_env.sh

# OFI Settings
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_CQ_SIZE=32768
export FI_CXI_OFLOW_BUF_SIZE=2097152
export FI_CXI_DEFAULT_VNI=0

SYSTEM="$SYS"
BLOCK_MB=$BLOCK_MB
EOF

        if [ "$SYS" == "lmcache-redis" ]; then
            cat <<EOF >> $TEMP_SLURM
export REDIS_PORT=16379
echo "=== Redis Setup ==="
mkdir -p benchmark/tmp/hosts_\$SLURM_JOB_ID
rm -f benchmark/tmp/hosts_\$SLURM_JOB_ID/redis_host
redis-server --port \$REDIS_PORT --bind 0.0.0.0 --protected-mode no --save "" &
REDIS_PID=\$!
sleep 15
echo \$(hostname) > benchmark/tmp/hosts_\$SLURM_JOB_ID/redis_host
EOF
        fi

        cat <<EOF >> $TEMP_SLURM
echo "=== Recovery Profiling: \$SYSTEM at $NODES nodes ==="
srun --overlap -N $NODES -n $NODES --ntasks-per-node=1 \\
    python3 benchmark/scripts/v12_recovery_profiling.py \\
    --system \$SYSTEM \\
    --block-size-mb \$BLOCK_MB \\
    --num-blocks 10
EOF

        if [ "$SYS" == "lmcache-redis" ]; then
            echo "kill \$REDIS_PID" >> $TEMP_SLURM
        fi

        echo "echo \"✅ Done\"" >> $TEMP_SLURM
        
        sbatch $TEMP_SLURM
        # rm $TEMP_SLURM
    done
done

# Extra: LMCache-Redis 32 nodes
NODES=32
SYS="lmcache-redis"
JOB_NAME="rec_${SYS}_${NODES}n"
TEMP_SLURM="benchmark/scripts/tmp_${JOB_NAME}.slurm"
cat <<EOF > $TEMP_SLURM
#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q $QUEUE
#SBATCH -t $TIME
#SBATCH -N $NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J $JOB_NAME
#SBATCH -o benchmark/logs/${JOB_NAME}_%j.out
#SBATCH -e benchmark/logs/${JOB_NAME}_%j.err

cd $REPO_DIR
source setup_env.sh

export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_STARTUP_CONNECT=1
export REDIS_PORT=16379

mkdir -p benchmark/tmp/hosts_\$SLURM_JOB_ID
rm -f benchmark/tmp/hosts_\$SLURM_JOB_ID/redis_host
redis-server --port \$REDIS_PORT --bind 0.0.0.0 --protected-mode no --save "" &
REDIS_PID=\$!
sleep 15
echo \$(hostname) > benchmark/tmp/hosts_\$SLURM_JOB_ID/redis_host

srun --overlap -N $NODES -n $NODES --ntasks-per-node=1 \\
    python3 benchmark/scripts/v12_recovery_profiling.py \\
    --system lmcache-redis \\
    --block-size-mb $BLOCK_MB \\
    --num-blocks 10

kill \$REDIS_PID
echo "✅ Done"
EOF

sbatch $TEMP_SLURM
