import gym

from config.base import BaseConfig
from core.model import ResModel
from core.util import DiscreteSupport
import place_env


class Config(BaseConfig):

    def __init__(
        self,
        training_steps: int = 40,
        pretrain_steps: int = 0,
        model_broadcast_interval: int = 1,
        model_save_interval: int = 10,
        num_sgd_iter: int = 10,
        clear_buffer_after_broadcast: bool = False,
        root_value_targets: bool = False,
        replay_buffer_size: int = 50000,
        demo_buffer_size: int = 0,
        batch_size: int = 128,
        lr: float = 1e-3,
        max_grad_norm: float = 5,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        c_init: float = 3.0,
        c_base: float = 19652,
        gamma: float = 0.997,
        frame_stack: int = 5,
        max_reward_return: bool = False,
        hash_nodes: bool = False,
        root_dirichlet_alpha: float = 1.5,
        root_exploration_fraction: float = 0.25,
        num_simulations: int = 50,
        num_envs_per_worker: int = 4,
        min_num_episodes_per_worker: int = 4,
        # min_num_episodes_per_worker: int = 8,
        use_dirichlet: bool = True,
        test_use_dirichlet: bool = False,
        value_support: DiscreteSupport = DiscreteSupport(0, 22, 1.0),
        value_transform: bool = True,
        env_seed: int = None,
        log_dir: str = None,
    ):
        super().__init__(
            training_steps,
            pretrain_steps,
            model_broadcast_interval,
            num_sgd_iter,
            clear_buffer_after_broadcast,
            root_value_targets,
            replay_buffer_size,
            demo_buffer_size,
            batch_size,
            lr,
            max_grad_norm,
            weight_decay,
            momentum,
            c_init,
            c_base,
            gamma,
            frame_stack,
            max_reward_return,
            hash_nodes,
            root_dirichlet_alpha,
            root_exploration_fraction,
            num_simulations,
            num_envs_per_worker,
            min_num_episodes_per_worker,
            use_dirichlet,
            test_use_dirichlet,
            value_support,
            value_transform,
            env_seed,
        )
        self.log_dir = log_dir

    def init_model(self, device, amp):
        obs_shape = (self.obs_shape[0] * self.frame_stack,) + self.obs_shape[1:]
        num_act = self.action_shape

        model = ResModel(self, obs_shape, num_act, device, amp)
        model.to(device)
        return model

    def env_creator(self, log_dir=None):
        if log_dir is None:
            log_dir = self.log_dir
        return gym.make("Place-v0", log_dir=log_dir)
