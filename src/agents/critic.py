"""
Q-network critic for actor–critic agents.

Approximates q(s, a); trained with TD error. Target network (separate copy)
is updated softly in the agent, not inside this module.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Critic(nn.Module):
    """
    Q-network: two hidden layers, concat(state, action) -> scalar Q.
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
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
