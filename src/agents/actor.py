import torch.nn as nn
from torch import Tensor


class GaussianActor(nn.Module):
    """
    Maps state s to Gaussian distribution pi(a|s) = N(mu(s), sigma(s)).
    Actions are squashed through tanh to satisfy action bounds.
    Used by both ActorCriticAgent and ReinforceAgent.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int) -> None:
        """
        Args:
            state_dim:  dimensionality of state vector (3 + K).
            action_dim: dimensionality of action (2).
            hidden_dim: number of units in hidden layers.
        """
        pass

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute mean and log-std of action distribution.

        Returns:
            mean, log_std — both shape (batch, action_dim).
        """
        pass

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Sample action via reparametrisation: a = tanh(mu + sigma*eps).

        Returns:
            action, log_prob (with tanh correction).
        """
        pass
