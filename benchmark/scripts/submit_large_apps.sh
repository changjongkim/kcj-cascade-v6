#!/bin/bash

# Configuration - Scaling up to "Real Lustre Load" sizes
NODES=(8 4 2 1)
APPS=("checkpoint" "ensemble")

mkdir -p benchmark/logs benchmark/tmp

for app in "${APPS[@]}"; do
    for n in "${NODES[@]}"; do
        QOS="regular"  # Use regular queue for larger datasets
        if [ $n -le 2 ]; then QOS="regular"; fi
        
        JOB_NAME="v6_${app}_n${n}_large"
        OUT_FILE="benchmark/logs/${app}_n${n}_large_%j.out"
        ERR_FILE="benchmark/logs/${app}_n${n}_large_%j.err"
        
        # Large Scale Parameters
        if [ "$app" == "checkpoint" ]; then
            # 1GB fields x 5 fields = 5GB per rank per checkpoint
            # 8 ranks = 40GB per checkpoint. 2 checkpoints = 80GB raw I/O.
            ARGS="--mode checkpoint --field-size 1024 --num-fields 5 --timesteps 10 --ckpt-interval 5 --delta-pct 5 --systems Cascade,HDF5,POSIX"
            PY_SCRIPT="benchmark/scripts/v6_app_checkpoint.py"
        else
            # 512MB blocks. IC = 10GB shared. Member = 5GB x 5 steps = 25GB per member.
            # 8 members = 200GB+ total cluster data.
            ARGS="--block-size 512 --ic-blocks 20 --member-blocks 5 --evolution-steps 5 --exchange-rounds 3 --systems Cascade,HDF5,POSIX"
            PY_SCRIPT="benchmark/scripts/v6_app_ensemble.py"
        fi
        
        cat <<EOT > "benchmark/scripts/submit_${app}_n${n}_large.slurm"
#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q $QOS
#SBATCH -t 00:30:00
#SBATCH -N $n
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J $JOB_NAME
#SBATCH -o $OUT_FILE
#SBATCH -e $ERR_FILE

cd /pscratch/sd/s/sgkim/kcj/Cascade-kcj
source setup_env.sh
export CASCADE_BUILD_DIR=/pscratch/sd/s/sgkim/kcj/Cascade-kcj/cascade_Code/cpp/build_cascade_cpp

echo "Running LARGE SCALE $app with $n nodes ($QOS queue)"
srun python3 $PY_SCRIPT $ARGS
EOT
        
        echo "Submitting $JOB_NAME..."
        sbatch "benchmark/scripts/submit_${app}_n${n}_large.slurm"
    done
done
