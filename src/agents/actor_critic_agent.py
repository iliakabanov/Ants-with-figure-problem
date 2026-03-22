"""
Off-policy actor–critic with SAC-style entropy regularisation.

- Replay buffer with uniform random minibatches.
- Q-critic with a slowly moving target network (Polyak averaging, ``tau``).
- Soft Bellman backup: y = r + γ (Q_target(s',a') − α log π(a'|s')).
- Actor loss: E[ α log π(a|s) − Q(s,a) ] (same α as ``config.entropy_coef``).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from src.utils.config import MAZE_ACTION_DIM

from .actor import GaussianActor
from .critic import Critic
from .replay_buffer import ReplayBuffer


class ActorCriticAgent:
    """
    Actor maximises Q(s, a); critic fits Bellman targets from Q_target.

    Args:
        state_dim:  observation size.
        action_dim: action size (e.g. 3 for MazeEnv).
        config:     Config with lr_*, gamma, tau, batch_size, buffer_capacity, warmup_steps, hidden_dim.
        device:     torch device.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self._state_dim = state_dim
        self._action_dim = action_dim

        hd = config.hidden_dim
        self.actor = GaussianActor(state_dim, action_dim, hd).to(device)
        self.critic = Critic(state_dim, action_dim, hd).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        if action_dim != MAZE_ACTION_DIM:
            raise ValueError(
                f"ActorCriticAgent expects action_dim={MAZE_ACTION_DIM} "
                f"(fx_body, fy_body, delta_theta_deg), got {action_dim}."
            )
        # After tanh: same component order as MazeEnv.action_space Box bounds.
        self.action_scale = torch.tensor(
            [config.max_thrust, config.max_thrust, config.max_delta_theta],
            dtype=torch.float32,
            device=device,
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.shape[0] != self._state_dim:
            raise ValueError(f"Expected state length {self._state_dim}, got {s.shape[0]}.")
        state_tensor = torch.as_tensor(s, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)

        action = action * self.action_scale
        out = action.squeeze(0).cpu().numpy()
        if out.shape != (MAZE_ACTION_DIM,):
            raise RuntimeError(
                f"Actor must return action shape ({MAZE_ACTION_DIM},), got {out.shape}."
            )
        return out

    def _soft_update_critic_target(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(self) -> dict[str, float]:
        cfg = self.config
        if len(self.replay_buffer) < max(cfg.batch_size, cfg.warmup_steps):
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "mean_log_prob": 0.0,
            }

        alpha = cfg.entropy_coef
        batch = self.replay_buffer.sample(cfg.batch_size)
        s = batch["state"]
        a = batch["action"]
        r = batch["reward"]
        s_next = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            a_next, log_prob_next = self.actor.sample(s_next)
            a_next_scaled = a_next * self.action_scale
            q_next = self.critic_target(s_next, a_next_scaled)
            v_soft_next = q_next - alpha * log_prob_next
            y = r + cfg.gamma * (1.0 - done) * v_soft_next

        q = self.critic(s, a)
        critic_loss = 0.5 * ((q - y) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a_new, log_prob = self.actor.sample(s)
        a_new_scaled = a_new * self.action_scale
        q_pi = self.critic(s, a_new_scaled)
        actor_loss = (alpha * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_critic_target()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_log_prob": log_prob.mean().item(),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "critic_target" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target"])
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())
