"""
Gaussian actor network shared by ActorCriticAgent and ReinforceAgent.

The actor maps a state s to a Gaussian distribution over actions:
    pi(a | s) = N(mu_theta(s), sigma_theta(s))

Actions are squashed through tanh to satisfy the action bounds:
    a = tanh(mu + sigma * eps),  eps ~ N(0, I)
"""

import torch
import torch.nn as nn
from torch import Tensor


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    """
    Stochastic Gaussian actor: one hidden layer + mean / log_std heads.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        z = mean + std * eps
        action = torch.tanh(z)
        log_prob_gaussian = -0.5 * (
            ((z - mean) / std) ** 2 + 2 * log_std + torch.log(torch.tensor(2 * 3.14159265))
        )
        log_prob = log_prob_gaussian - torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
