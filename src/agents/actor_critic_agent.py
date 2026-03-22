"""
Actor-Critic agent for the figure-navigation task.

Training follows the critic learning lecture notes:
  - Critic is trained by minimising the TD-error over the last M+1
    transitions (the full replay buffer), with a stop-gradient bootstrap
    target. No target network is needed.
  - Actor is trained by maximising the critic's Q-value estimate,
    using the reparametrisation trick to flow gradients through the critic.

Update order each step:
  1. Push transition into replay buffer (last M+1 transitions).
  2. Compute TD target y = r + gamma * q(s', a')  [stop gradient].
  3. Critic loss: L_crit = 0.5 * mean((q(s,a) - y)^2) over all M+1 transitions.
  4. Update critic weights omega.
  5. Actor loss: L_actor = -mean(q(s, actor(s))) over all M+1 transitions.
  6. Update actor weights theta.
"""

import numpy as np
import torch
from torch import Tensor

from .actor import GaussianActor
from .critic import Critic
from .replay_buffer import ReplayBuffer


class ActorCriticAgent:
    """
    Off-policy Actor-Critic agent with continuous action space.

    The critic learns q*(s, a) via TD-error minimisation over the last
    M+1 transitions, with stop-gradient targets (no target network).
    The actor learns to maximise q*(s, a) by flowing gradients through
    the critic via the reparametrisation trick.

    Args:
        state_dim:  dimensionality of the state vector (3 + K).
        action_dim: dimensionality of the action vector (2).
        config:     dataclass with all hyperparameters (see utils/config.py).
        device:     torch device to run on.
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

        # --- Networks ---
        self.actor = GaussianActor(
            state_dim, action_dim, config.hidden_dim
        ).to(device)

        self.critic = Critic(
            state_dim, action_dim, config.hidden_dim
        ).to(device)

        # --- Optimisers ---
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic
        )

        # --- Replay buffer ---
        # Stores only the last M+1 transitions, matching the lecture.
        # No random sampling — all M+1 transitions are used every update.
        self.replay_buffer = ReplayBuffer(
            M=config.M,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        # Scale factor: tanh maps to [-1, 1], rescale to action bounds.
        self.action_scale = torch.FloatTensor([
            config.max_delta_theta,
            config.max_thrust,
        ]).to(device)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select an action for the given state.

        During training (deterministic=False), samples from pi(a|s)
        using the reparametrisation trick.
        During evaluation (deterministic=True), uses the mean action
        mu_theta(s) — no randomness.

        Args:
            state:         state vector, shape (state_dim,).
            deterministic: if True, use mean action (no exploration).

        Returns:
            action: numpy array of shape (action_dim,),
                    scaled to actual action bounds.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)

        action = action * self.action_scale
        return action.squeeze(0).cpu().numpy()

    def update(self) -> dict[str, float]:
        """
        Perform one gradient update step for both critic and actor.

        Called once per environment step during training, as soon as
        the replay buffer holds at least one transition (no warmup).
        Uses all M+1 stored transitions, matching the lecture loss:

            L_crit(omega | R_t) = 0.5 * sum_{k=0}^{M} (delta(d_{t-k}))^2

        Update logic:
          1. Fetch all M+1 transitions from the replay buffer.
          2. Compute TD target with stop-gradient:
               y = r + gamma * (1 - done) * q(s', a')
             where a' ~ pi(. | s') is sampled from the current actor.
          3. Critic update: minimise 0.5 * mean((q(s, a) - y)^2).
          4. Actor update: minimise -mean(q(s, actor(s))).
             Gradient flows: theta -> action -> critic -> loss -> theta.
             Critic weights omega are NOT updated in this step.

        Returns:
            metrics: dict with 'critic_loss' and 'actor_loss' (float).
        """
        # Fetch all M+1 transitions from the replay buffer
        batch  = self.replay_buffer.get_all()
        s      = batch["state"]
        a      = batch["action"]
        r      = batch["reward"]
        s_next = batch["next_state"]
        done   = batch["done"]

        # ── Step 1: Compute TD target (stop gradient) ──────────────────
        # The target y is fixed as a constant for this update step.
        # This stabilises training without needing a target network,
        # following the approach from the lecture.
        with torch.no_grad():
            a_next, _ = self.actor.sample(s_next)
            a_next_scaled = a_next * self.action_scale
            q_next = self.critic(s_next, a_next_scaled)
            y = r + self.config.gamma * (1.0 - done) * q_next

        # ── Step 2: Update critic ──────────────────────────────────────
        # Minimise TD-error: L = 0.5 * mean((q(s, a) - y)^2)
        # Gradient flows only through q(s, a), not through y.
        q = self.critic(s, a)
        critic_loss = 0.5 * ((q - y) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Step 3: Update actor ───────────────────────────────────────
        # Maximise E[q(s, actor(s))] by minimising -E[q(s, actor(s))].
        # Gradient flows: theta -> a_new -> critic -> loss -> theta.
        # Critic weights omega are NOT updated here.
        a_new, _ = self.actor.sample(s)
        a_new_scaled = a_new * self.action_scale
        actor_loss = -self.critic(s, a_new_scaled).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
        }

    def save(self, path: str) -> None:
        """
        Save actor and critic weights plus optimiser states to disk.

        Args:
            path: file path for the checkpoint
                  (e.g. 'artifacts/checkpoints/ac_ep1000.pt').
        """
        torch.save({
            "actor":            self.actor.state_dict(),
            "critic":           self.critic.state_dict(),
            "actor_optimizer":  self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """
        Load actor and critic weights plus optimiser states from disk.

        Args:
            path: file path of the checkpoint to load.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
