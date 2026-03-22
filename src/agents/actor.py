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
    Stochastic Gaussian actor network.

    Takes a state vector and outputs the mean and log-std of a Gaussian
    distribution over actions. Actions are squashed through tanh so they
    lie in [-1, 1], then rescaled to the actual action bounds by the agent.

    Args:
        state_dim:  dimensionality of the state vector (3 + K ray distances).
        action_dim: dimensionality of the action vector (2: delta_theta, f).
        hidden_dim: number of units in each hidden layer.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute mean and log-std of the action distribution.

        Args:
            state: state tensor of shape (batch, state_dim).

        Returns:
            mean:    action mean,    shape (batch, action_dim).
            log_std: log-std,        shape (batch, action_dim).
        """
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Sample an action using the reparametrisation trick.

        The reparametrisation trick separates the randomness from the
        parameters:
            a = tanh(mu_theta(s) + sigma_theta(s) * eps),  eps ~ N(0, I)

        This makes 'a' a differentiable function of theta, so gradients
        can flow from the critic back into the actor during actor updates.

        Args:
            state: state tensor of shape (batch, state_dim).

        Returns:
            action:   sampled and tanh-squashed action, shape (batch, action_dim).
            log_prob: log pi(a | s) with tanh correction, shape (batch, 1).
                      Used by REINFORCE for the policy gradient.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample using reparametrisation: z = mu + sigma * eps
        eps = torch.randn_like(mean)
        z = mean + std * eps

        # Squash through tanh
        action = torch.tanh(z)

        # Compute log probability with tanh correction:
        # log pi(a|s) = log N(z; mu, sigma) - sum log(1 - tanh(z)^2)
        log_prob_gaussian = -0.5 * (((z - mean) / std) ** 2 + 2 * log_std + torch.log(torch.tensor(2 * 3.14159265)))
        log_prob = log_prob_gaussian - torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

