import torch.nn as nn
from torch import Tensor


class Critic(nn.Module):
    """Q-networks Q_1, Q_2. Used by Actor-Critic only."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int) -> None:
        pass

    def forward(self, state: Tensor,
                action: Tensor) -> tuple[Tensor, Tensor]:
        """Return Q_1(s,a), Q_2(s,a)."""
        pass

    def min_q(self, state: Tensor, action: Tensor) -> Tensor:
        """Return min(Q_1, Q_2) for actor loss."""
        pass
