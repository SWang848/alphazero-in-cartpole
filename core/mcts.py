from copy import deepcopy
import random
import math
import numpy as np
import torch

from core.util import MinMaxStats, sample_gumbel


class Node:
    def __init__(self, config, action, num_actions):
        self.config = config
        self.action = action
        self.num_actions = num_actions
        self.parent_traversed = None

        self.reward = None
        self.obs = None
        self.env_state = None
        self.info = None
        self.terminal = False
        self.expanded = False
        self.child_logits = None

        self.num_visits = 0
        self.value_sum = 0

        self.children = {}

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
        for _, child in self.children.items():  # Update min-max stats
            child_value = child.mean_value()
            if child.num_visits > 0:
                min_max_stats.update(
                    accu([child.reward, self.config.gamma * child_value])
                )
            # if child is not visited, update min-max stats with mean_q
            else:
                if mean_q is not None:
                    min_max_stats.update(mean_q)

        for _, child in self.children.items():  # Calculate child values
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
        best_found = {"hpwl": float("inf"), "reward": None, "state": None}
        min_max_stats = [MinMaxStats() for _ in range(roots.root_num)]
        selected_actions = []
        root_q_values = []
        for i in range(roots.root_num):
            remaining_sim_budget = self.config.num_simulations
            m = self.config.m_top
            root = roots.roots[i]
            window = mcts_windows[i]
            min_max_stat = min_max_stats[i]
            logits = root.child_logits
            gumbel_noise = sample_gumbel(logits.shape)
            gumbel_logits = logits + gumbel_noise
            top_m_indices = np.argsort(gumbel_logits)[-m:][::-1]
            selected_logits = gumbel_logits[top_m_indices]
            selected_children = [root.children[action] for action in top_m_indices]

            while remaining_sim_budget > 0:

                # Search for each selected child
                if m == 2 or m == 3:
                    num_sims_per_action = max(1, math.ceil(remaining_sim_budget / m))
                else:
                    num_sims_per_action = max(
                        1,
                        math.floor(
                            self.config.num_simulations
                            / (m * np.log2(self.config.m_top))
                        ),
                    )
                for child in selected_children:
                    child_best_found = self.search(
                        root, window, min_max_stat, num_sims_per_action, child
                    )
                    if child_best_found["hpwl"] < best_found["hpwl"]:
                        best_found = child_best_found

                max_visit_among_children = max(
                    child.num_visits for child in selected_children
                )
                sigma_logits = selected_logits.copy()
                child_values = root.child_values(min_max_stat)
                for i, child in enumerate(selected_children):
                    sigma_transform = (
                        (self.config.c_visit + max_visit_among_children)
                        * self.config.c_scale
                        * child_values[child.action]
                    )
                    sigma_logits[i] = selected_logits[i] + sigma_transform

                remaining_sim_budget -= num_sims_per_action * m
                m = max(1, math.floor(m / 2))
                top_m_indices = np.argsort(sigma_logits)[-m:][::-1]
                selected_children = [selected_children[i] for i in top_m_indices]
                selected_logits = [selected_logits[i] for i in top_m_indices]
            selected_actions.append(selected_children[0].action)
            root_q_values.append(root.child_values(min_max_stat, root.mean_q(0)))

        return selected_actions, roots.get_values(), root_q_values, best_found

    def search(
        self,
        root,
        mcts_window,
        min_max_stat,
        num_simulations,
        designed_search_node=None,
    ):
        self.env.reset()
        best_found = {"hpwl": float("inf"), "reward": None, "state": None}

        for simulation_index in range(num_simulations):
            window = deepcopy(mcts_window)
            trajectory = self.traverse(root, window, min_max_stat, designed_search_node)

            from_node = trajectory[-2]
            to_node = trajectory[-1]

            self.env = self.env.set_state(from_node.env_state)
            obs, reward, done, truncated, info = self.env.step(to_node.action)
            window.add(
                obs["board_image"],
                self.env.get_state(),
                reward,
                to_node.action,
                info,
            )

            if info["hpwl"] < best_found["hpwl"]:
                best_found = {
                    "hpwl": info["hpwl"],
                    "reward": reward,
                    "state": self.env.get_state(),
                }
            prior, value, _ = self.model.compute_priors_and_values([window])

            self.backpropagate(
                to_node, window, value[0], prior[0], done, info, min_max_stat
            )

        return best_found

    def traverse(self, root, mcts_window, min_max_stats, designed_search_node=None):
        trajectory = []
        node = root
        parent_q = 0

        trajectory.append(node)

        while node.expanded:
            mean_q = node.mean_q(parent_q)
            if node == root and designed_search_node is not None:
                best_child = designed_search_node
            else:
                best_child = node.best_child(min_max_stats, mean_q)
            best_child.parent_traversed = node
            if (
                best_child.expanded
            ):  # We can not do node.obs at the beginning of the loop, because the root node is already inside the sliding window
                mcts_window.add(
                    best_child.obs,
                    best_child.env_state,
                    best_child.reward,
                    best_child.action,
                    best_child.info,
                )
            node = best_child
            trajectory.append(node)
        return trajectory

    def backpropagate(
        self, leaf_node, mcts_window, value, prior, terminal, info, min_max_stat
    ):
        node: Node = leaf_node  # Take vals for current leaf_node
        o = mcts_window.latest_obs()
        reward = mcts_window.rewards[0]
        state = mcts_window.env_state

        node.expand(o, reward, terminal, info, state, prior)
        accu = max if self.config.max_reward_return else sum
        if terminal:
            value = 0.0
        while True:
            node.value_sum += value
            # node.value_sum = (node.num_visits * node.value_sum + value) / (node.num_visits + 1)
            node.num_visits += 1
            min_max_stat.update(node.mean_value())

            if node.parent_traversed is None:
                break
            reward = node.reward
            value = accu([reward, self.config.gamma * value])
            parent_node = node.parent_traversed
            node.parent_traversed = None
            node = parent_node
