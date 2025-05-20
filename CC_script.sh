#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=30G
#SBATCH --time=72:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/shang8/scratch/slurm_out/%A.out
#SBATCH --mail-user=shang8@ualberta.ca
#SBATCH --mail-type=ALL

echo $1 #lr
echo $2 #c_init
echo $3 #c_scale
# echo $4 #seed

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline # log offline
export WANDB__SERVICE_WAIT=300
export RAY_memory_usage_threshold=0.95
export VTR_ROOT=/home/shang8/scratch/vtr-verilog-to-routing
export results=$SLURM_TMPDIR/results
cp -R /home/shang8/scratch/alphazero-in-cartpole/data $SLURM_TMPDIR/data
export data=$SLURM_TMPDIR/data

export HEAD_NODE=$(ip route get 1 | awk '{print $7; exit}')
export BASE_PORT=10000
export RAY_PORT=$(python3 -c "
import socket
with socket.socket() as s:
    s.bind(('', 0))
    port = s.getsockname()[1]
    print(port)
")

module load python/3.10
module load cuda
source /home/shang8/scratch/MCTS_env/bin/activate
wandb offline

export sub_dir="seed_$((RANDOM % 10000))"
ray start --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT --include-dashboard=false --metrics-export-port=0 --num-cpus=12 --num-gpus=1

PYTHONUNBUFFERED=1 python3 -u main.py --cc --wandb --group_name c5b_gumbel_cc_classic_kl --amp --env Classic-v0 --seed 0 --num_rollout_workers 10 --num_cpus_per_worker 1.2 --num_envs_per_worker 20 --num_gpus_per_worker 0.1 --min_num_episodes_per_worker 20 --num_target_blocks 5 --num_simulations 20 --training_steps 10 --c_init $2 --lr $1 --value_support_min -1 --value_support_max 0 --value_support_delta 0.1 --m_top 4 --c_visit 8 --c_scale $3

cp -r $results/* /home/shang8/scratch/alphazero-in-cartpole/results/