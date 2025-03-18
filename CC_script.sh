#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=60G
#SBATCH --time=50:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/shang8/scratch/slurm_out/%A.out
#SBATCH --mail-user=shang8@ualberta.ca
#SBATCH --mail-type=ALL

# echo $1 # c_init
# echo $2 # lr
echo $1 # seed

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline # log offline
export VTR_ROOT=/home/shang8/scratch/vtr-verilog-to-routing
export results=$SLURM_TMPDIR/results
cp -R /home/shang8/scratch/alphazero-in-cartpole/data $SLURM_TMPDIR/data
export data=$SLURM_TMPDIR/data
export HEAD_NODE=$(hostname)
export RAY_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

module load python/3.10
module load cuda
source /home/shang8/scratch/MCTS_env/bin/activate
wandb offline

ray start --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT --num-cpus=12 --num-gpus=1 --block &
sleep 20

PYTHONUNBUFFERED=1 python3 -u main.py --cc --wandb --amp  --group_name c15b_fixed_init --env Swap-v0 --seed 0 --num_rollout_workers 10 --num_cpus_per_worker 1.2 --num_envs_per_worker 30 --num_gpus_per_worker 0.1 --min_num_episodes_per_worker 30 --num_target_blocks 15 --num_simulations 150 --training_steps 80 --c_init 2.5 --lr 1e-2 --value_support_min -10 --value_support_max 0 --value_support_delta 1
cp -r $results/* /home/shang8/scratch/alphazero-in-cartpole/results/
