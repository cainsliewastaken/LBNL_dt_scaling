#!/bin/bash -l
#SBATCH -t 12:00:00
#SBATCH -C gpu&hbm80g
#SBATCH -A m4790
#SBATCH --qos premium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J ViT_multistep_dt_0.03
#SBATCH --output ViT_multistep_dt_0.03.out
#SBATCH --error ViT_multistep_dt_0.03.err
#SBATCH --mem=224GB

module load conda
conda activate cainslie_env

# for DDP
export MASTER_ADDR=$(hostname)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cmd="python -u -m train_2d_turb_ddp.py --config config.yaml"

# move wandb cache to scratch
export WANDB_CACHE_DIR=/pscratch/sd/c/cainslie/wandb_cache


set -x
srun -l \
    bash -c "
    source export_DDP_vars.sh 
    $cmd
    " 