#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=10G
#SBATCH --time=60:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/shang8/scratch/slurm_out/%A.out
#SBATCH --mail-user=shang8@ualberta.ca
#SBATCH --mail-type=ALL


# echo $1 #entropy_coeff
# echo $2 #lr_a
# echo $3 #lr_c
echo $1 #place_order

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline # log offline
export WANDB__SERVICE_WAIT=300
export VTR_ROOT=/home/shang8/scratch/vtr-verilog-to-routing
export results=$SLURM_TMPDIR/results
cp -R /home/shang8/scratch/alphazero-in-cartpole/data $SLURM_TMPDIR/data
export data=$SLURM_TMPDIR/data

module load python/3.10
module load cuda
source /home/shang8/scratch/MCTS_env/bin/activate
wandb offline

# c15b classic MDP; ppo
# python3 -u ppo_main.py --cc --wandb --group_name c15b_classic_PPO --env Classic-v0 --seed 0 \
#         --num_target_blocks 15 --num_envs 8 --lr_a $2 --lr_c $3 --entropy_coef $1 \
#         --total_timesteps 256000 --rollout_size 64 --mini_batch_size 32

# c15b swap MDP; ppo
# python3 -u ppo_main.py --cc --wandb --group_name c15b_swap_PPO --env Swap-v0 --seed 0 \
#         --num_target_blocks 15 --num_envs 8 --lr_a $2 --lr_c $3 --entropy_coef $1 \
#         --total_timesteps 256000 --rollout_size 64 --mini_batch_size 32

PYTHONUNBUFFERED=1 python3 -u ppo_main.py --wandb --cc --group_name c5b_ppo_RL4RS --env Classic-v0 --seed 0 --num_target_blocks 5 --num_envs 10 --lr_a 5e-4 --lr_c 5e-4 --entropy_coef 5e-3 --max_training_time 600 --rollout_size 512 --mini_batch_size 128 --place_order $1