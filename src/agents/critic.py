"""
Q-network critic for the ActorCriticAgent.

The critic approximates the action-value function q*(s, a).
It is trained using TD-error minimisation with a stop-gradient on the
target, following the approach from the critic learning lecture notes.

No target network is used. Instead, the TD target y is computed with
torch.no_grad() to stabilise training:
    y = r + gamma * q_omega(s', a')   [stop gradient]
    L = 0.5 * (q_omega(s, a) - y)^2
"""

import torch
import torch.nn as nn
from torch import Tensor


class Critic(nn.Module):
    """
    Q-network that approximates q*(s, a).

    Takes a (state, action) pair and outputs a scalar Q-value estimate.
    Trained by minimising the TD-error between the current Q-value and
    a stop-gradient bootstrap target.

    Args:
        state_dim:  dimensionality of the state vector (3 + K).
        action_dim: dimensionality of the action vector (2).
        hidden_dim: number of units in each hidden layer.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Compute Q-value estimate for a batch of (state, action) pairs.

        Args:
            state:  state tensor of shape (batch, state_dim).
            action: action tensor of shape (batch, action_dim).

        Returns:
            q: Q-value estimates of shape (batch, 1).
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
