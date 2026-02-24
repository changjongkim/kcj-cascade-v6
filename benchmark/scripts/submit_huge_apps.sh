#!/bin/bash

# Configuration - HUGE SCALE: 600GB+ real Lustre load
NODES=(8 4 2 1)
APPS=("checkpoint" "ensemble")

mkdir -p benchmark/logs benchmark/tmp

for app in "${APPS[@]}"; do
    for n in "${NODES[@]}"; do
        QOS="regular"
        
        JOB_NAME="v6_${app}_n${n}_huge"
        OUT_FILE="benchmark/logs/${app}_n${n}_huge_%j.out"
        ERR_FILE="benchmark/logs/${app}_n${n}_huge_%j.err"
        
        # HUGE Scale Parameters (Target: ~600GB per app)
        if [ "$app" == "checkpoint" ]; then
            # 2GB fields x 5 fields = 10GB per rank per checkpoint
            # 8 ranks = 80GB per checkpoint. 8 checkpoints = 640GB total.
            # Timesteps 16, Interval 2.
            ARGS="--mode checkpoint --field-size 2048 --num-fields 5 --timesteps 16 --ckpt-interval 2 --delta-pct 5 --systems Cascade,HDF5,POSIX"
            PY_SCRIPT="benchmark/scripts/v6_app_checkpoint.py"
        else
            # 1GB blocks. IC = 100GB shared. Member = 60GB (10 blocks x 6 steps).
            # 8 members = 100 + 480 = 580GB total.
            ARGS="--block-size 1024 --ic-blocks 100 --member-blocks 10 --evolution-steps 6 --exchange-rounds 3 --systems Cascade,HDF5,POSIX"
            PY_SCRIPT="benchmark/scripts/v6_app_ensemble.py"
        fi
        
        cat <<EOT > "benchmark/scripts/submit_${app}_n${n}_huge.slurm"
#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q $QOS
#SBATCH -t 01:00:00
#SBATCH -N $n
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J $JOB_NAME
#SBATCH -o $OUT_FILE
#SBATCH -e $ERR_FILE

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Running HUGE SCALE $app with $n nodes ($QOS queue)"
srun python3 $PY_SCRIPT $ARGS
EOT
        
        echo "Submitting $JOB_NAME..."
        sbatch "benchmark/scripts/submit_${app}_n${n}_huge.slurm"
    done
done
