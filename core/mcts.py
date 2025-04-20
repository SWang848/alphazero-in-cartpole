import math
from typing import Optional, Dict
import numpy as np
import ray
from copy import deepcopy

from config.base import BaseConfig
from core.util import MinMaxStats, sample_gumbel


@ray.remote
class GumbelSearchWorker:

    def __init__(
        self,
        config: BaseConfig,
    ):
        self.config = config
        self.env = config.env_creator(num_target_blocks=config.num_target_blocks)

    def process_batch(
        self,
        root_nodes,
        designed_children_nodes,
        mcts_windows,
        min_max_stats,
        best_found,
    ):

        # Initialize storage for results
        leaf_results = []  # Store tuples of (leaf_node, window, done, info, env_index)
        windows = deepcopy(mcts_windows)
        self.env.reset()

        # traverse trees
        trajectories = []
        for i, (root, child, window, stats) in enumerate(
            zip(root_nodes, designed_children_nodes, windows, min_max_stats)
        ):
            node = root
            parent_q = 0
            trajectory = [node]

            while node.expanded:
                mean_q = node.mean_q(parent_q)
                if node == root:
                    best_child = child
                else:
                    best_child = node.best_child(stats, mean_q)
                best_child.parent_traversed = node

                if best_child.expanded:
                    window.add(
                        best_child.obs,
                        best_child.env_state,
                        best_child.reward,
                        best_child.action,
                        best_child.info,
                    )
                node = best_child
                trajectory.append(node)
            trajectories.append(trajectory)

        # Process each trajectory
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) == 1:
                continue

            from_node = trajectory[-2]
            to_node = trajectory[-1]

            self.env.set_state(from_node.env_state)
            obs, reward, done, truncated, info = self.env.step(to_node.action)

            if info["hpwl"] < best_found["hpwl"]:
                best_found = {
                    "hpwl": info["hpwl"],
                    "reward": reward,
                    "state": self.env.get_state(),
                }

            # Update window and collect results
            windows[i].add(
                obs["board_image"],
                self.env.get_state(),
                reward,
                to_node.action,
                info,
            )
            leaf_results.append((to_node, windows[i], done, info, i))

        return leaf_results, best_found


class Node:
    def __init__(self, config, action, num_actions):
        self.config = config
        self.action = action
        self.num_actions = num_actions
        self.parent_traversed: Optional[Node] = None

        self.reward: Optional[float] = None
        self.obs: Optional[np.ndarray] = None
        self.env_state: Optional[any] = None
        self.info: Optional[dict] = None
        self.terminal: bool = False
        self.expanded: bool = False
        self.child_logits: Optional[np.ndarray] = None

        self.num_visits: int = 0
        self.value_sum: float = 0.0

        self.children: Dict[int, Node] = {}

        self.child_priors = None

    def expand(
        self,
        obs,
        reward,
        terminal,
        info,
        state,
        priors: np.ndarray,
        logits: np.ndarray = None,
    ):
        self.obs = obs
        self.reward = reward
        self.terminal = terminal
        self.env_state = state
        self.info = info
        self.child_logits = logits
        if terminal:
            return

        self.child_priors = priors
        for i in range(self.num_actions):
            self.children[i] = Node(self.config, i, self.num_actions)

        self.expanded = True

    def add_child_logits(self, logits):
        self.child_logits = logits

    def add_exploration_noise(self, noise, exploration_fraction):
        self.child_priors = np.where(
            self.info["action_mask"],
            self.child_priors * (1 - exploration_fraction)
            + noise * exploration_fraction,
            0.0,
        )
        # self.child_priors = np.where(self.child_priors != 0,
        #                              self.child_priors * (1 - exploration_fraction) + noise * exploration_fraction,
        #                              self.child_priors)

    def child_number_visits(self):
        return np.array([child.num_visits for _, child in self.children.items()])

    def child_values(self, min_max_stats, mean_q=None):
        values = []
        accu = max if self.config.max_reward_return else sum
        for _, child in self.children.items():
            child_value = child.mean_value()
            if child.num_visits > 0:
                child_value = min_max_stats.normalize(
                    accu([child.reward, self.config.gamma * child_value])
                )
            else:
                if mean_q is not None:
                    child_value = min_max_stats.normalize(mean_q)
                else:
                    child_value = 0.0
            # clip child_value to [0, 1] to avoid out of range
            child_value = min(max(child_value, 0.0), 1.0)
            values.append(child_value)
        return np.array(values)  # Return normalized values

    def mean_value(self):
        return self.value_sum / self.num_visits if self.num_visits > 0 else 0.0

    def mean_q(self, parent_q):
        # see EfficientZero p.18, 'mean_q' function
        total_q = 0
        total_visits = 0
        for _, child in self.children.items():
            if child.num_visits > 0:
                total_q += child.reward + self.config.gamma * child.mean_value()
                total_visits += 1

        if self.parent_traversed is None and total_visits > 0:
            mean_q = total_q / total_visits
        else:
            mean_q = (total_q + parent_q) / (total_visits + 1)

        return mean_q

    def get_child(self, action):
        return self.children[action]

    def puct_scores(self, min_max_stats, mean_q=None):
        # See: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf
        # p. 17, Section "Search"
        c_base = self.config.c_base
        c_init = self.config.c_init
        c_term = np.log((1 + self.num_visits + c_base) / c_base) + c_init
        visit_term = np.sqrt(self.num_visits) / (self.child_number_visits() + 1)

        prior_score = c_term * visit_term * self.child_priors
        value_score = self.child_values(min_max_stats, mean_q)
        return value_score + prior_score

    def best_action(self, min_max_stats: MinMaxStats, mean_q):
        score = self.puct_scores(min_max_stats, mean_q)
        masked_score = np.where(self.info["action_mask"], score, -np.inf)
        # masked_score = np.where(self.child_priors != 0, score, -np.inf)
        max_val = np.max(masked_score)
        action = np.random.choice(np.argwhere(masked_score == max_val).flatten())
        return action

    def best_child(self, min_max_stats, mean_q):
        return self.children[self.best_action(min_max_stats, mean_q)]


class BatchTree:
    def __init__(self, root_num, num_actions, config):
        self.root_num = root_num
        self.action_shape = num_actions
        self.config = config

        self.roots = []
        for _ in range(root_num):
            root = Node(self.config, None, num_actions)
            self.roots.append(root)

        self.node_hash_tables = [{} for _ in range(root_num)]

    def prepare(self, mcts_windows, priors, logits):
        for i in range(self.root_num):
            prior = priors[i]
            state = mcts_windows[i].env_state
            root = self.roots[i]
            root.num_visits += 1
            info = mcts_windows[i].infos[0]
            logit = logits[i]
            if not root.expanded and not root.terminal:
                root.expand(mcts_windows[i].obs, None, False, info, state, prior, logit)
            elif not root.terminal and root.child_logits is None:
                root.add_child_logits(logit)

    def backpropagate(
        self, leaf_nodes, mcts_windows, values, priors, terminals, infos, min_max_stats
    ):
        for i in range(len(leaf_nodes)):
            node: Node = leaf_nodes[i]  # Take vals for current leaf_node
            o = mcts_windows[i].latest_obs()
            reward = mcts_windows[i].rewards[0]
            state = mcts_windows[i].env_state
            value = values[i]
            prior = priors[i]
            terminal = terminals[i]
            info = infos[i]

            # Expand the leaf node
            # If it's a terminal node, the `expand` call will return without expansion
            node.expand(o, reward, terminal, info, state, prior)

            # Define return accumulation function. For vanilla RL, we use R_t = r_t + gamma R_(t+1)
            # If `config.max_reward_return`, R_t = max(r_t, R_(t_1))
            accu = max if self.config.max_reward_return else sum
            if terminal:
                value = 0.0
            while True:
                node.value_sum += value
                # node.value_sum = (node.num_visits * node.value_sum + value) / (node.num_visits + 1)
                node.num_visits += 1
                min_max_stats[i].update(node.mean_value())

                if node.parent_traversed is None:
                    break
                reward = node.reward
                value = accu([reward, self.config.gamma * value])
                parent_node = node.parent_traversed
                node.parent_traversed = None
                node = parent_node

    def apply_actions(self, actions):
        for i in range(self.root_num):
            if actions[i] is None:
                continue

            root = self.roots[i]
            if root.terminal:
                continue

            action = actions[i]
            new_root = root.get_child(action)
            new_root.parent = None
            self.roots[i] = new_root

    def get_distributions(self):
        dists = []
        for root in self.roots:
            dists.append(root.child_number_visits())
        return dists

    def get_values(self):
        values = []
        for root in self.roots:
            values.append(root.mean_value())
        return values

    def get_children_values(self, min_max_stats, mean_q):
        values = []
        for root in self.roots:
            values.append(root.child_values(min_max_stats, mean_q))
        return values

    def clear(self):
        self.roots = None
        self.node_hash_tables = [{} for _ in range(self.root_num)]


class MCTS:
    def __init__(
        self,
        config,
        model,
    ):
        self.config = config
        self.model = model
        self._search_workers = []

        # Maximum number of parallel workers for Gumbel search
        self.max_parallel_searches = self.config.max_parallel_searches

    def gumbel_squential_halving_search(self, roots, mcts_windows):
        # Ensure this instance has enough search workers
        while len(self._search_workers) < self.max_parallel_searches:
            # Create workers if they don't exist for this instance
            worker = GumbelSearchWorker.options(num_cpus=0.125).remote(self.config)
            self._search_workers.append(worker)

        # Initialize best found for this specific search
        best_found = {"hpwl": float("inf"), "reward": None, "state": None}
        min_max_stats = [MinMaxStats() for _ in range(roots.root_num)]
        remaining_sim_budget = self.config.num_simulations
        m = self.config.m_top

        # Get initial logits and apply Gumbel noise
        batch_logits = np.stack(
            [root.child_logits for root in roots.roots]
        )  # (num_root, num_actions)
        gumbel_noise = sample_gumbel(batch_logits.shape)  # (num_root, num_actions)
        gumbel_logits = batch_logits + gumbel_noise  # (num_root, num_actions)

        # Select initial top m actions
        top_m_indices = np.argsort(gumbel_logits, axis=1)[:, -m:][
            :, ::-1
        ]  # (num_root, m)
        selected_logits = np.take_along_axis(
            gumbel_logits, top_m_indices, axis=1
        )  # (num_root, m)
        selected_children = [
            [roots.roots[i].children[action] for action in actions]
            for i, actions in enumerate(top_m_indices)
        ]

        max_batch_size = min(m, self.max_parallel_searches)

        while remaining_sim_budget > 0:
            if m <= 3:
                num_sims_per_action = max(1, math.ceil(remaining_sim_budget / m))
            else:
                num_sims_per_action = max(
                    1,
                    math.floor(
                        self.config.num_simulations / (m * np.log2(self.config.m_top))
                    ),
                )

            for _ in range(num_sims_per_action):
                for batch_start in range(0, m, max_batch_size):
                    batch_end = min(m, batch_start + max_batch_size)
                    futures = []

                    # Pass references to workers instead of serializing for each task
                    mcts_windows_ref = ray.put(mcts_windows)
                    roots_ref = ray.put(roots.roots)

                    for i in range(batch_start, batch_end):
                        batch_children = np.array(selected_children)[:, i]
                        worker_idx = i % len(
                            self._search_workers
                        )  # Use instance variable

                        futures.append(
                            self._search_workers[worker_idx].process_batch.remote(
                                roots_ref,
                                batch_children,
                                mcts_windows_ref,
                                min_max_stats,
                                best_found,
                            )
                        )

                    batch_results = ray.get(
                        futures
                    )  # len(batch_results) = max_batch_size
                    all_leaf_nodes = []
                    all_windows = []
                    all_dones = []
                    all_infos = []
                    all_env_indices = []

                    for search_worker_result in batch_results:
                        leaf_results, worker_best_found = (
                            search_worker_result  # len(leaf_results) = num_root
                        )

                        if worker_best_found["hpwl"] < best_found["hpwl"]:
                            best_found = worker_best_found

                        for leaf_node, window, done, info, env_index in leaf_results:
                            all_leaf_nodes.append(leaf_node)
                            all_windows.append(window)
                            all_dones.append(done)
                            all_infos.append(info)
                            all_env_indices.append(env_index)

                    priors, values, _ = self.model.compute_priors_and_values(
                        all_windows
                    )

                    for i in range(len(all_leaf_nodes)):

                        self.backpropagate(
                            all_leaf_nodes[i],
                            all_windows[i],
                            values[i],
                            priors[i],
                            all_dones[i],
                            all_infos[i],
                            min_max_stats[all_env_indices[i]],
                        )

            # Update selection scores for all roots
            max_visits = np.array(
                [
                    max(child.num_visits for child in children)
                    for children in selected_children
                ]
            )  # (num_root,)
            children_values = np.array(
                [
                    roots.roots[j].child_values(min_max_stats[j])
                    for j in range(roots.root_num)
                ]
            )  # (num_root, num_actions)

            actions = np.array(
                [[child.action for child in children] for children in selected_children]
            )  # (num_root, m)
            sigma_transforms = (
                (self.config.c_visit + max_visits[:, None])
                * self.config.c_scale
                * np.take_along_axis(children_values, actions, axis=1)
            )
            sigma_logits = selected_logits + sigma_transforms  # (num_root, m)

            # Update remaining budget and halve number of candidates
            remaining_sim_budget -= num_sims_per_action * m
            m = max(1, m // 2)

            # Select top m candidates for next iteration
            top_m_indices = np.argsort(sigma_logits, axis=1)[:, -m:][
                :, ::-1
            ]  # (num_root, m)
            selected_children = [
                [children[j] for j in indices]
                for children, indices in zip(selected_children, top_m_indices)
            ]
            selected_logits = np.take_along_axis(
                sigma_logits, top_m_indices, axis=1
            )  # (num_root, m)

        # Get final selected actions and values
        selected_actions = [children[0].action for children in selected_children]
        root_q_values = [
            roots.roots[i].child_values(min_max_stats[i], roots.roots[i].mean_q(0))
            for i in range(roots.root_num)
        ]

        return selected_actions, roots.get_values(), root_q_values, best_found

    def backpropagate(self, leaf_node, window, value, prior, done, info, min_max_stats):
        leaf_node.expand(
            window.latest_obs(), window.rewards[0], done, info, window.env_state, prior
        )

        accu = max if self.config.max_reward_return else sum
        if done:
            value = 0.0  # Terminal state value is 0

        current_node = leaf_node
        propagated_value = value
        while current_node is not None:
            current_node.value_sum += propagated_value
            current_node.num_visits += 1
            parent_node = current_node.parent_traversed

            if parent_node is not None:  # If not the root node
                reward = current_node.reward
                qsa = accu([reward, self.config.gamma * current_node.mean_value()])
                min_max_stats.update(qsa)
                propagated_value = accu([reward, self.config.gamma * propagated_value])

            else:  # This is the root node
                min_max_stats.update(current_node.mean_value())

            if parent_node is not None:
                current_node.parent_traversed = None
            current_node = parent_node
