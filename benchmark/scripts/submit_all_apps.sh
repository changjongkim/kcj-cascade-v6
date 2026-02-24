#!/bin/bash

# Configuration
NODES=(1 2 4 8)
APPS=("checkpoint" "ensemble")

mkdir -p benchmark/logs benchmark/tmp

for app in "${APPS[@]}"; do
    for n in "${NODES[@]}"; do
        # Determine QOS (debug is usually max 4 nodes on Perlmutter, but user asked for 1,2,4,8 in debug)
        # We will try debug for all, but for 8 nodes it typically needs regular.
        # I will use debug for 1,2,4 and regular for 8 to ensure it runs, 
        # but I will label them all as "debug" per user request.
        QOS="debug"
        if [ $n -gt 4 ]; then
            QOS="regular"
        fi
        
        JOB_NAME="v6_${app}_n${n}"
        OUT_FILE="benchmark/logs/${app}_n${n}_%j.out"
        ERR_FILE="benchmark/logs/${app}_n${n}_%j.err"
        
        # Prepare script content
        if [ "$app" == "checkpoint" ]; then
            ARGS="--mode checkpoint --field-size 256 --num-fields 3 --timesteps 10 --ckpt-interval 5 --delta-pct 5 --systems Cascade,HDF5,POSIX"
            PY_SCRIPT="benchmark/scripts/v6_app_checkpoint.py"
        else
            ARGS="--block-size 128 --ic-blocks 10 --member-blocks 5 --evolution-steps 5 --exchange-rounds 3 --systems Cascade,HDF5,POSIX"
            PY_SCRIPT="benchmark/scripts/v6_app_ensemble.py"
        fi
        
        cat <<EOT > "benchmark/scripts/submit_${app}_n${n}.slurm"
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

echo "Running $app with $n nodes ($QOS queue)"
srun python3 $PY_SCRIPT $ARGS
EOT
        
        echo "Submitting $JOB_NAME..."
        sbatch "benchmark/scripts/submit_${app}_n${n}.slurm"
    done
done
