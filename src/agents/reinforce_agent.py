"""
REINFORCE policy gradient agent with pluggable baseline.

REINFORCE (Williams, 1992) is an on-policy Monte Carlo policy gradient
algorithm. It collects a full episode, then computes the return G_t at
each step and updates the policy to increase the probability of actions
that led to high returns.

Update rule:
    grad J(theta) = E[ sum_t grad log pi(a_t|s_t) * (G_t - b(s_t)) ]

where b(s_t) is a baseline that reduces variance without introducing bias.

Key differences from ActorCriticAgent:
  - On-policy: uses transitions from the current policy only.
  - No replay buffer: the episode buffer is cleared after each update.
  - No critic network: returns G_t are estimated via Monte Carlo.
  - Update happens once per episode (not every step).
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .actor import GaussianActor
from .baselines import BaseBaseline, ZeroBaseline


class ReinforceAgent:
    """
    REINFORCE agent with Monte Carlo returns and a pluggable baseline.

    The agent collects full episodes by calling store_transition() at
    each step. At the end of the episode, update() computes the returns,
    subtracts the baseline, and performs one gradient update on the actor.

    Args:
        state_dim:  dimensionality of the state vector (3 + K).
        action_dim: dimensionality of the action vector (2).
        config:     dataclass with all hyperparameters (see utils/config.py).
        device:     torch device to run on.
        baseline:   BaseBaseline instance. Defaults to ZeroBaseline().
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config,
        device: torch.device,
        baseline: BaseBaseline | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.baseline = baseline if baseline is not None else ZeroBaseline()

        # --- Actor network ---
        self.actor = GaussianActor(
            state_dim, action_dim, config.hidden_dim
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_reinforce
        )

        # Scale factor: tanh maps to [-1, 1], rescale to action bounds.
        # Action order: (fx_body, fy_body, delta_theta_deg)
        self.action_scale = torch.FloatTensor([
            config.max_thrust,
            config.max_thrust,
            config.max_delta_theta,
        ]).to(device)

        # --- Episode buffer ---
        # Stores transitions collected during the current episode.
        # Cleared after each call to update().
        self._states:    list[np.ndarray] = []
        self._actions:   list[np.ndarray] = []
        self._rewards:   list[float]      = []
        self._log_probs: list[Tensor]     = []

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Tensor]:
        """
        Select an action for the given state.

        During training (deterministic=False), samples from pi(a|s)
        using the reparametrisation trick and returns the log probability
        for use in the policy gradient update.

        During evaluation (deterministic=True), uses the mean action
        mu_theta(s) — no randomness, no log probability needed.

        Args:
            state:         state vector, shape (state_dim,).
            deterministic: if True, use mean action (no exploration).

        Returns:
            action:   numpy array of shape (action_dim,),
                      scaled to actual action bounds.
            log_prob: log pi(a|s) tensor, shape (1, 1).
                      Returns None if deterministic=True.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if deterministic:
            with torch.no_grad():
                mean, _ = self.actor.forward(state_tensor)
                action = torch.tanh(mean) * self.action_scale
            return action.squeeze(0).cpu().numpy(), None

        # Sample with reparametrisation — keep graph for update()
        action_tanh, log_prob = self.actor.sample(state_tensor)
        action = action_tanh * self.action_scale
        return action.detach().squeeze(0).cpu().numpy(), log_prob

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: Tensor,
    ) -> None:
        """
        Store one transition from the current episode.

        Must be called after every env.step() during episode collection.
        The log_prob tensor must still be attached to the computation
        graph (do not detach it) so that gradients can flow during update().

        Args:
            state:    state vector s_t, shape (state_dim,).
            action:   action vector a_t, shape (action_dim,).
            reward:   scalar reward r_t = r(s_t, a_t).
            log_prob: log pi(a_t | s_t) tensor from select_action(),
                      shape (1, 1), still attached to the computation graph.
        """
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._log_probs.append(log_prob)

    def update(self) -> dict[str, float]:
        """
        Policy gradient update at the end of an episode.

        Steps:
          1. Compute Monte Carlo returns:
               G_t = sum_{k=t}^{T} gamma^(k-t) * r_k
          2. For each step t, query baseline: b_t = baseline.estimate(s_t).
          3. Compute advantages: A_t = G_t - b_t.
          4. Compute policy gradient loss:
               L = -sum_t log pi(a_t|s_t) * A_t
             (negative because we maximise J but minimise L)
          5. Backpropagate and update actor weights theta.
          6. Update baseline with observed (states, returns) pairs.
          7. Clear episode buffer.

        Returns:
            metrics: dict with 'policy_loss' (float), 'mean_return' (float),
                     'mean_advantage' (float).
        """
        T = len(self._rewards)

        # ── Step 1: Compute Monte Carlo returns ────────────────────────
        # G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        # Computed by iterating backwards through the episode.
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.config.gamma * G
            returns.insert(0, G)

        # ── Step 2 & 3: Subtract baseline and compute advantages ───────
        advantages = []
        for t in range(T):
            b_t = self.baseline.estimate(self._states[t])
            advantages.append(returns[t] - b_t)

        # ── Step 4: Policy gradient loss ───────────────────────────────
        # L = -sum_t log pi(a_t|s_t) * A_t
        log_probs_t = torch.stack([lp.squeeze() for lp in self._log_probs])
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        loss = -(log_probs_t * advantages_t).sum() / T

        # ── Step 5: Gradient update ────────────────────────────────────
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        # ── Step 6: Update baseline ────────────────────────────────────
        self.baseline.update(self._states, returns)

        # ── Step 7: Clear episode buffer ───────────────────────────────
        metrics = {
            "policy_loss":    loss.item(),
            "mean_return":    float(np.mean(returns)),
            "mean_advantage": float(np.mean(advantages)),
        }
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()

        return metrics

    def save(self, path: str) -> None:
        """
        Save actor weights, optimiser state, and baseline state to disk.

        Args:
            path: file path for the checkpoint
                  (e.g. 'artifacts/checkpoints/reinforce_ep1000.pt').
        """
        torch.save({
            "actor":           self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "baseline":        self.baseline.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """
        Load actor weights, optimiser state, and baseline state from disk.

        Args:
            path: file path of the checkpoint to load.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.baseline.load_state_dict(checkpoint["baseline"])
