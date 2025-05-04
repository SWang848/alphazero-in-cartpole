import math
from typing import Optional, Dict
import numpy as np
from copy import deepcopy

from core.util import MinMaxStats, sample_gumbel


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

    def traverse(self, mcts_windows, min_max_stats, designed_search_nodes=None):
        trajectories = []
        for i in range(self.root_num):
            node = self.roots[i]
            parent_q = 0
            trajectories.append([node])

            while node.expanded:
                mean_q = node.mean_q(parent_q)
                if (
                    designed_search_nodes is not None and node == self.roots[i]
                ):  # the designed search node is only working for the root node
                    best_child = designed_search_nodes[i]
                else:
                    best_child = node.best_child(min_max_stats[i], mean_q)
                best_child.parent_traversed = node
                if (
                    best_child.expanded
                ):  # We can not do node.obs at the beginning of the loop, because the root node is already inside the sliding window
                    mcts_windows[i].add(
                        best_child.obs,
                        best_child.env_state,
                        best_child.reward,
                        best_child.action,
                        best_child.info,
                    )
                node = best_child
                trajectories[-1].append(node)
        return trajectories

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
                if node.parent_traversed is None:
                    break
                reward = node.reward
                min_max_stats[i].update(reward + self.config.gamma * node.mean_value())
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
        self.env = config.env_creator(num_target_blocks=config.num_target_blocks)

    def gumbel_squential_halving_search(self, roots, mcts_windows):
        self.env.reset()
        best_found = {"hpwl": float("inf"), "reward": None, "state": None}
        min_max_stats = [MinMaxStats() for _ in range(roots.root_num)]
        remaining_sim_budget = self.config.num_simulations
        m = self.config.m_top

        batch_logits = np.stack([root.child_logits for root in roots.roots])
        gumbel_noise = sample_gumbel(batch_logits.shape)
        gumbel_logits = batch_logits + gumbel_noise

        # Select initial top m actions more efficiently
        top_m_indices = np.argsort(gumbel_logits, axis=1)[:, -m:][:, ::-1]
        selected_logits = np.take_along_axis(gumbel_logits, top_m_indices, axis=1)

        # Pre-compute child nodes for faster lookup
        roots_children = [root.children for root in roots.roots]
        selected_children = [
            [roots_children[i][action] for action in actions]
            for i, actions in enumerate(top_m_indices)
        ]

        while remaining_sim_budget > 0:
            if m <= 3:
                num_sims_per_action = max(1, math.ceil(remaining_sim_budget / m))
            else:
                log2_m_top = np.log2(self.config.m_top)
                num_sims_per_action = max(
                    1, math.floor(self.config.num_simulations / (m * log2_m_top))
                )

            for _ in range(num_sims_per_action):
                for i in range(m):
                    batch_children = np.array(selected_children)[:, i]
                    windows = deepcopy(mcts_windows)
                    trajectories = roots.traverse(
                        windows, min_max_stats, batch_children
                    )

                    leaf_nodes = []
                    dones = []
                    infos = []

                    # Process each trajectory
                    for env_index in range(roots.root_num):
                        trajectory = trajectories[env_index]
                        if len(trajectory) == 1:
                            dones.append(True)
                            continue

                        from_node = trajectory[-2]
                        to_node = trajectory[-1]

                        # Take environment step
                        self.env = self.env.set_state(from_node.env_state)
                        obs, reward, done, truncated, info = self.env.step(
                            to_node.action
                        )

                        # Update best found solution
                        if info["hpwl"] < best_found["hpwl"]:
                            best_found = {
                                "hpwl": info["hpwl"],
                                "reward": reward,
                                "state": self.env.get_state(),
                            }

                        # Update window and collect results
                        windows[env_index].add(
                            obs["board_image"],
                            self.env.get_state(),
                            reward,
                            to_node.action,
                            info,
                        )
                        leaf_nodes.append(to_node)
                        dones.append(done)
                        infos.append(info)

                    priors, values, _ = self.model.compute_priors_and_values(windows)
                    roots.backpropagate(
                        leaf_nodes, windows, values, priors, dones, infos, min_max_stats
                    )

            max_visits = np.array(
                [
                    max(child.num_visits for child in children)
                    for children in selected_children
                ]
            )

            children_values = np.array(
                [
                    roots.roots[j].child_values(
                        min_max_stats[j], roots.roots[j].mean_q(0)
                    )
                    for j in range(roots.root_num)
                ]
            )  # (num_root, num_actions)
            # Calculate actions array
            actions = np.array(
                [[child.action for child in children] for children in selected_children]
            )

            c_visit_term = (
                self.config.c_visit + max_visits[:, None]
            ) * self.config.c_scale
            action_values = np.take_along_axis(children_values, actions, axis=1)
            sigma_transforms = c_visit_term * action_values
            sigma_logits = selected_logits + sigma_transforms

            remaining_sim_budget -= num_sims_per_action * m
            m = max(1, m // 2)

            top_m_indices = np.argsort(sigma_logits, axis=1)[:, -m:][:, ::-1]

            new_selected_children = []
            selected_children = [
                [children[j] for j in indices]
                for children, indices in zip(selected_children, top_m_indices)
            ]
            selected_logits = np.take_along_axis(sigma_logits, top_m_indices, axis=1)

        # Get final selected actions and values
        selected_actions = [children[0].action for children in selected_children]
        root_q_values = [
            roots.roots[i].child_values(min_max_stats[i], roots.roots[i].mean_q(0))
            for i in range(roots.root_num)
        ]

        return selected_actions, roots.get_values(), root_q_values, best_found
