#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J dc_repro_cas_opt_1n
#SBATCH -o benchmark/logs/v16_deepcam_repro_cas_opt_1n_%j.out

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source ml_workspace/setup_ml_isolated.sh

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $RANDOM % 20000)

export DATA_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/deepcam_dummy_data_512gb"
export OUT_DIR="/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/results/deepcam_repro_cas_opt_1n_${SLURM_JOB_ID}"

echo "--------------------------------------------------------"
echo "🚀 DeepCAM V16 Repro Benchmark (cascade_opt 1 Nodes)"
echo "--------------------------------------------------------"

srun python ml_workspace/scripts/v16_app_deepcam_proxy_opt.py \
       --use_cascade --no_aggregated_io --cascade_lustre_path /pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cas_opt_store_${SLURM_JOB_ID} \
       --wireup_method "mpi" \
       --run_tag "deepcam_repro_cas_opt_1n" \
       --data_dir_prefix ${DATA_DIR} \
       --output_dir ${OUT_DIR} \
       --model_prefix "segmentation" \
       --optimizer "Adam" \
       --start_lr 0.0055 \
       --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \
       --lr_warmup_steps 400 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --logging_frequency 10 \
       --save_frequency 0 \
       --max_epochs 1 \
       --max_inter_threads 4 \
       --batchnorm_group_size 1 \
       --local_batch_size 2

echo "✅ Done: dc_repro_cas_opt_1n"
