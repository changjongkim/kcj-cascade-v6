#!/bin/bash

# Configuration
NODES=(2)
SYSTEMS=("cascade_noagg" "hdf5" "pdc" "lmcache_disk" "llm_gpu")
QUEUE="regular"
TIME_LIMIT="00:40:00"

mkdir -p benchmark/slurm

for N in "${NODES[@]}"; do
    for SYS in "${SYSTEMS[@]}"; do
        CURRENT_QUEUE=$QUEUE
        CURRENT_TIME=$TIME_LIMIT
        if [ "$N" == "1" ] || [ "$N" == "2" ]; then
            CURRENT_QUEUE="debug"
            CURRENT_TIME="00:30:00"
        fi
        
        JOB_NAME="dc_repro_${SYS}_${N}n"
        SLURM_FILE="benchmark/slurm/${JOB_NAME}.slurm"
        OUT_LOG="benchmark/logs/v16_deepcam_repro_${SYS}_${N}n_%j.out"
        
        # Determine Python arguments based on system
        PY_ARGS=""
        if [ "$SYS" == "cascade_noagg" ]; then
            PY_ARGS="--use_cascade --no_aggregated_io --cascade_lustre_path /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cascade_noagg_store_\${SLURM_JOB_ID}"
        else
            PY_ARGS="--system ${SYS} --storage_path /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/${SYS}_store_\${SLURM_JOB_ID}"
        fi
        
        cat <<EOT > $SLURM_FILE
#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q $CURRENT_QUEUE
#SBATCH -t $CURRENT_TIME
#SBATCH -N $N
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J $JOB_NAME
#SBATCH -o $OUT_LOG

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source ml_workspace/setup_ml_isolated.sh

export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=\$(expr 10000 + \$RANDOM % 20000)

export DATA_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/deepcam_dummy_data_512gb"
export OUT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/results/deepcam_repro_${SYS}_${N}n_\${SLURM_JOB_ID}"

echo "--------------------------------------------------------"
echo "🚀 DeepCAM V16 Repro Benchmark ($SYS $N Nodes)"
echo "--------------------------------------------------------"

srun python ml_workspace/scripts/v16_app_deepcam_proxy.py \\
       $PY_ARGS \\
       --wireup_method "mpi" \\
       --run_tag "deepcam_repro_${SYS}_${N}n" \\
       --data_dir_prefix \${DATA_DIR} \\
       --output_dir \${OUT_DIR} \\
       --model_prefix "segmentation" \\
       --optimizer "Adam" \\
       --start_lr 0.0055 \\
       --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \\
       --lr_warmup_steps 400 \\
       --lr_warmup_factor 1. \\
       --weight_decay 1e-2 \\
       --logging_frequency 10 \\
       --save_frequency 0 \\
       --max_epochs 1 \\
       --max_inter_threads 4 \\
       --batchnorm_group_size 1 \\
       --local_batch_size 2

echo "✅ Done: $JOB_NAME"
EOT

        DEP_ARG=""
        if [ ! -z "$1" ]; then
            DEP_ARG="--dependency=afterok:$1"
        fi
        
        echo "Submitting $JOB_NAME with dependency $1..."
        sbatch $DEP_ARG $SLURM_FILE
    done
done
