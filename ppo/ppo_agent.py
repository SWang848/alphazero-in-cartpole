import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from ppo.model import Actor, Critic


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, masks=None, device="cpu"):
        self.original_logits = logits.clone()
        self.device = device
        self.masks = masks
        if masks is None:
            super(CategoricalMasked, self).__init__(probs=probs, logits=logits)
        else:
            logits = torch.where(self.masks, logits, torch.tensor(-float('1e8')).to(self.device))
            super(CategoricalMasked, self).__init__(probs=probs, logits=logits)

    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)


class PPO:
    def __init__(
        self,
        num_actions,
        observation_shape,
        args,
    ) -> None:
        self.device = args.device

        self.mini_batch_size = args.mini_batch_size
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.k_epochs = args.k_epochs
        self.entropy_coef = args.entropy_coef

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.num_envs = args.num_envs

        self.actor = Actor(
            obs_shape=observation_shape,
            num_act=num_actions,
            device=args.device
        ).to(self.device)
        self.critic = Critic(
            obs_shape=observation_shape,
            device=args.device
        ).to(self.device)
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_a, eps=1e-5
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_c, eps=1e-5
        )

    def choose_action(
        self,
        board_image: np.ndarray,
        action_mask: np.ndarray,
    ) -> (int, float): # type: ignore
        board_image = torch.tensor(board_image, dtype=torch.float32).to(self.device)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            logits = self.actor(board_image)

            dist = CategoricalMasked(
                logits=logits,
                masks=action_mask,
                device=self.device,
            )
            a = dist.sample()

            # use masked probs
            a_logprob = dist.log_prob(a)

            # use unmasked probs
            # dist = CategoricalMasked(logits=logits, masks=None, device=self.device)
            # a_logprob = dist.log_prob(a)

        return a.cpu().numpy(), a_logprob.cpu().numpy()

    def update(self, rollouts: object):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        old_approx_kl_epoch = 0
        approx_kl_epoch = 0
        clipfracs_epoch = []

        adv = []
        clipfracs = []

        with torch.no_grad():
            vs = self.critic(
                rollouts.board_image.reshape(
                    -1, *self.observation_shape
                )
            ).reshape(-1, self.num_envs)
            vs_ = self.critic(
                rollouts.board_image_.reshape(
                    -1, *self.observation_shape
                )
            ).reshape(-1, self.num_envs)

            lastgae = 0
            adv = torch.zeros_like(rollouts.rewards).to(self.device)
            for t in reversed(range(rollouts.rollout_size)):
                delta = (
                    rollouts.rewards[t]
                    + self.gamma * vs_[t] * (1.0 - rollouts.done[t])
                    - vs[t]
                )

                adv[t] = lastgae = delta + self.gamma * self.lamda * lastgae * (
                    1.0 - rollouts.done[t]
                )

            v_targets = adv + vs
            v_targets = v_targets.reshape(-1)
            adv = adv.reshape(-1)
            vs = vs.reshape(-1)
            vs_ = vs_.reshape(-1)

        for _ in range(self.k_epochs):
            for index in BatchSampler(
                SubsetRandomSampler(range(rollouts.rollout_size * self.num_envs)),
                batch_size=self.mini_batch_size,
                drop_last=False,
            ):
                logits = self.actor(
                    rollouts.board_image.reshape(
                        -1, *self.observation_shape
                    )[index]
                )

                # using masked policy to update
                dist_now = CategoricalMasked(
                    logits=logits,
                    masks=rollouts.action_mask.reshape(-1, self.num_actions)[index],
                    device=self.device,
                )
                dist_entropy = dist_now.entropy().view(-1, 1)
                a_logprob_now = dist_now.log_prob(rollouts.actions.reshape(-1)[index])

                logratio = a_logprob_now - rollouts.action_log_probs.reshape(-1)[index]
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = [
                        (((ratio - 1.0).abs() > self.epsilon).float().mean().item())
                    ]

                # entropy loss
                entropy_loss = dist_entropy.mean()

                # actor loss
                surr1 = ratio * adv[index]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * adv[index]
                )
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy_loss
                )

                self.optimizer_actor.zero_grad()
                # with torch.autograd.set_detect_anomaly(True):
                actor_loss.backward()
                # for name, param in self.actor.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: max grad {param.grad.max()}, min grad {param.grad.min()}")
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # critic loss
                v_s = self.critic(
                    rollouts.board_image.reshape(
                        -1, *self.observation_shape
                    )[index]
                )

                critic_loss = (v_s - v_targets[index]) ** 2
                critic_clipped = vs[index] + (v_s - vs[index]).clamp(
                    -self.epsilon, self.epsilon
                )
                critic_loss_clipped = (critic_clipped - v_targets[index]) ** 2
                critic_loss = 0.5 * torch.max(critic_loss, critic_loss_clipped).mean()

                self.optimizer_critic.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                value_loss_epoch += critic_loss.item()
                action_loss_epoch += actor_loss.item()
                dist_entropy_epoch += entropy_loss.item()
                old_approx_kl_epoch += old_approx_kl.item()
                approx_kl_epoch += approx_kl.item()
                clipfracs_epoch += clipfracs

        value_loss_epoch /= self.k_epochs
        action_loss_epoch /= self.k_epochs
        dist_entropy_epoch /= self.k_epochs
        old_approx_kl_epoch /= self.k_epochs
        approx_kl_epoch /= self.k_epochs
        clipfracs_epoch = np.mean(clipfracs_epoch)

        return (
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
            old_approx_kl_epoch,
            approx_kl_epoch,
            clipfracs_epoch,
        )
