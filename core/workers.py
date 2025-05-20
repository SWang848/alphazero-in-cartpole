from copy import deepcopy
import time

import numpy as np
import ray

from core.mcts import BatchTree, MCTS
from config.base import BaseConfig
from core.replay_buffer import TransitionBuffer, ReplayBuffer, MCTSRollingWindow
from core.storage import SharedStorage


class MCTSWorker:
    def __init__(
        self,
        config: BaseConfig,
        device: str,
        amp: bool,
        num_envs: int,
        use_dirichlet: bool,
        worker_id: int = 0,
        simulator: bool = False,
    ):
        self.config = config
        self.model = self.config.init_model(device, amp)
        self.model.eval()
        self.num_envs = num_envs
        self.use_dirichlet = use_dirichlet
        self.worker_id = worker_id

        self.envs = [
            config.env_creator(
                simulator=simulator,
                num_target_blocks=config.num_target_blocks,
            )
            for _ in range(self.num_envs)
        ]
        self.env_observation_space = self.envs[0].observation_space["board_image"]
        self.env_action_space = self.envs[0].action_space
        
        # Create a single MCTS instance for this worker to avoid recreating actors for each episode
        self.mcts = MCTS(self.config, self.model)

    def _reset_envs(self, env):
        """Reset environment with appropriate seed"""
        # in test, we set the seed larger than 2**25, so the test init never show in the training
        if self.config.non_fixed_init:
            seed = np.random.randint(0, 2**20) + self.worker_id * 1000
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset(seed=self.config.seed)
        return obs, info

    def collect(self):
        roots = BatchTree(
            self.num_envs, self.envs[0].action_space.n, self.config
        )  # Prepare datastructures
        transition_buffers = [TransitionBuffer() for _ in range(self.num_envs)]
        mcts_windows = [
            MCTSRollingWindow(self.config.obs_shape, self.config.frame_stack)
            for _ in range(self.num_envs)
        ]
        finished = [False] * self.num_envs

        for i, env in enumerate(self.envs):
            obs, info = self._reset_envs(env)
            mcts_windows[i].add(
                obs=obs["board_image"],
                env_state=env.get_state(),
                reward=None,
                action=None,
                info=info,
            )

        current_best_found = {"hpwl": float("inf"), "reward": None, "state": None}
        while not all(finished):
            # Prepare roots
            priors, values, roots_logits = self.model.compute_priors_and_values(mcts_windows)

            roots.prepare(mcts_windows, priors, values, roots_logits)
            windows = deepcopy(mcts_windows)

            selected_actions, roots_values, roots_q_values, best_found = (
                self.mcts.gumbel_squential_halving_search(roots, windows)
            )

            if current_best_found["hpwl"] > best_found["hpwl"]:
                current_best_found = best_found

            # Execute action sampled from MCTS policy
            for env_index, selected_action in enumerate(selected_actions):

                obs, reward, done, truncated, info = self.envs[env_index].step(
                    selected_action
                )  # Apply action

                value_target = roots_values[env_index]

                transition_buffers[env_index].add_one(  # Add experience to data storage
                    mcts_windows[
                        env_index
                    ].latest_obs(),  # The observation the action is based upon (vs. `obs`, which is the observation the action generated)
                    selected_action,
                    reward,
                    done,
                    info,
                    [],
                    value_target,
                    roots_q_values[env_index],
                    mcts_windows[env_index].env_state,
                    1.0,  # TODO
                )

                mcts_windows[env_index].add(
                    obs["board_image"],
                    self.envs[env_index].get_state(),
                    reward=reward,
                    action=selected_action,
                    info=info,
                )  # Update rolling window for frame stacking

                if done:
                    finished[env_index] = True
                    if (
                        not self.config.root_value_targets
                    ):  # Overwrite root values calculated during MCTS search with actual trajectory state returns
                        transition_buffers[env_index].augment_value_targets(
                            max if self.config.max_reward_return else sum,
                            gamma=self.config.gamma,
                        )

            roots.apply_actions(
                selected_actions
            )  # Move the tree roots to the new nodes of actions taken

        roots.clear()

        return transition_buffers, current_best_found

    def evaluate(self):
        roots = BatchTree(
            self.num_envs, self.envs[0].action_space.n, self.config
        )  # Prepare datastructures
        # Use the pre-created MCTS instance instead of creating a new one
        # mcts = MCTS(self.config, self.model)
        mcts_windows = [
            MCTSRollingWindow(self.config.obs_shape, self.config.frame_stack)
            for _ in range(self.num_envs)
        ]

        for i, env in enumerate(self.envs):
            obs, info = self._reset_envs(env)
            mcts_windows[i].add(
                obs=obs["board_image"],
                env_state=env.get_state(),
                reward=None,
                action=None,
                info=info,
            )

        # Prepare roots
        priors, values, _ = self.model.compute_priors_and_values(
            mcts_windows
        )  # Compute priors and values for nodes to be expanded

        noises = None  # Inject noise into priors if configured
        if self.use_dirichlet:
            noises = [
                np.random.dirichlet(
                    [self.config.root_dirichlet_alpha] * self.env_action_space.n
                ).astype(np.float32)
                for _ in range(self.num_envs)
            ]
        roots.prepare(
            mcts_windows, self.config.root_exploration_fraction, priors, noises
        )

        windows = deepcopy(mcts_windows)
        _, _, best_found = self.mcts.search(roots, windows)  # Do MCTS search

        roots.clear()
        return best_found


@ray.remote
class RolloutWorker(MCTSWorker):
    def __init__(
        self,
        config: BaseConfig,
        device: str,
        amp: bool,
        replay_buffer: ReplayBuffer,
        storage: SharedStorage,
        worker_id: int = 0,
    ):
        num_envs = config.num_envs_per_worker
        use_dirichlet = config.use_dirichlet
        super().__init__(config, device, amp, num_envs, use_dirichlet, worker_id)

        self.replay_buffer = replay_buffer
        self.storage = storage

    def run(self):

        while True:  # Wait for start signal
            if not ray.get(self.storage.get_start_signal.remote()):
                time.sleep(1)
                continue
            break

        collect_update_step = -1
        while True:
            # Check if training finished
            update_step = ray.get(self.storage.get_counter.remote())
            # if update_step >= self.config.training_steps:
            #     time.sleep(30)
            #     break

            if collect_update_step == update_step:
                time.sleep(5)
                continue

            # Update weights
            model_weights = ray.get(self.storage.get_weights.remote())
            self.model.set_weights(model_weights)

            # Collect data
            whole_transition_buffers = []
            episode_best_found = {"hpwl": float("inf"), "reward": None, "state": None}
            while (
                len(whole_transition_buffers) < self.config.min_num_episodes_per_worker
            ):
                transition_buffers, best_found = self.collect()
                if episode_best_found["hpwl"] > best_found["hpwl"]:
                    episode_best_found = best_found

                # set the global best found
                if (
                    ray.get(self.storage.get_best_found.remote())["hpwl"]
                    > best_found["hpwl"]
                ):
                    self.storage.set_best_found.remote(best_found)
                whole_transition_buffers.extend(transition_buffers)

            # Add episode data to replay buffer and stats to storage
            stats = TransitionBuffer.compute_stats_buffers(whole_transition_buffers)
            wandb_stats = TransitionBuffer.compute_wandb_buffers(
                whole_transition_buffers, episode_best_found["hpwl"]
            )
            self.storage.add_wandb_logs.remote(wandb_stats)
            self.storage.add_rollout_worker_logs.remote(stats)
            self.replay_buffer.add.remote(whole_transition_buffers)

            collect_update_step = update_step
            self.storage.incr_workers_finished.remote()


@ray.remote
class TestWorker(MCTSWorker):
    def __init__(
        self,
        config: BaseConfig,
        device: str,
        amp: bool,
        worker_id: int = 0,
        simulator: bool = False,
    ):
        num_envs = config.num_envs_per_worker
        use_dirichlet = config.test_use_dirichlet
        super().__init__(
            config, device, amp, num_envs, use_dirichlet, worker_id, simulator
        )

        self.stats = None
        self.evaluation_stats = None
        self.best_found = {"hpwl": float("inf"), "reward": None, "state": None}

    def run(self, model_weights, num_episodes):
        whole_transition_buffers = []

        self.model.set_weights(model_weights)

        # Collect data
        while len(whole_transition_buffers) < num_episodes:
            transition_buffers, best_found = self.collect()
            if best_found["hpwl"] < self.best_found["hpwl"]:
                self.best_found = best_found
            whole_transition_buffers.extend(transition_buffers)

        # Compute and store stats
        self.stats = TransitionBuffer.compute_stats_buffers(transition_buffers)
        self.evaluation_stats = TransitionBuffer.compute_evaluation_buffers(
            transition_buffers
        )

    def get_stats(self):
        return self.stats, self.evaluation_stats, self.best_found


@ray.remote
class EvaluateWorker(MCTSWorker):
    def __init__(
        self,
        config: BaseConfig,
        device: str,
        amp: bool,
        worker_id: int = 0,
        simulator: bool = False,
    ):
        num_envs = config.num_envs_per_worker
        use_dirichlet = config.test_use_dirichlet
        super().__init__(
            config, device, amp, num_envs, use_dirichlet, worker_id, simulator
        )

        self.stats = None
        self.evaluation_stats = None
        self.best_found = {"hpwl": float("inf"), "reward": None, "state": None}

    def run(self, model_weights):
        self.model.set_weights(model_weights)

        # Evalute policy
        self.best_found = self.evaluate()

    def get_stats(self):
        return self.best_found


@ray.remote
class DemonstrationWorker:
    def __init__(
        self,
        config: BaseConfig,
        replay_buffer: ReplayBuffer,
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.env = config.demonstrator_env_creator()

    def collect(self):
        transition_buffer = TransitionBuffer()
        demo_traj = self.config.collect_demonstration(self.env)

        for i in range(len(demo_traj["obs"])):
            obs = demo_traj["obs"][i]
            rew = demo_traj["rewards"][i]
            done = demo_traj["dones"][i]
            info = demo_traj["infos"][i]
            env_state = demo_traj["env_states"][i]
            priority = 1.0

            action = demo_traj["actions"][i]
            mcts_policy = np.zeros(self.env.action_space.n)
            mcts_policy[action] = 1.0

            transition_buffer.add_one(
                obs,
                rew,
                done,
                info,
                mcts_policy,
                None,  # Will be calculated afterwards
                env_state,
                priority,
            )

        transition_buffer.augment_value_targets(
            max if self.config.max_reward_return else sum
        )  # Computes state returns based on rewards
        return transition_buffer

    def run(self):
        # Wait until start signal
        while ray.get(self.replay_buffer.size.remote()) < self.config.demo_buffer_size:
            # while self.replay_buffer.size() < self.config.demo_buffer_size:
            transition_buffer = self.collect()
            self.replay_buffer.add.remote(transition_buffer)
            # self.replay_buffer.add(transition_buffer)
            if (
                ray.get(self.replay_buffer.size.remote())
                >= self.config.demo_buffer_size
            ):
                print("Collection done")
