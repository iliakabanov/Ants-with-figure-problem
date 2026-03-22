"""
Uniform replay buffer for standard off-policy actor–critic.

Stores transitions in a circular buffer; ``sample(batch_size)`` draws
a random minibatch without replacement when possible.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """
    Fixed-capacity FIFO replay with uniform random sampling.

    Args:
        capacity:   maximum number of transitions.
        state_dim:  state vector length.
        action_dim: action vector length.
        device:     torch device for sampled batches.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        n = min(batch_size, self.size)
        if n <= 0:
            raise ValueError("Replay buffer is empty.")
        idx = np.random.choice(self.size, size=n, replace=False)
        return {
            "state": torch.as_tensor(self.states[idx], device=self.device),
            "action": torch.as_tensor(self.actions[idx], device=self.device),
            "reward": torch.as_tensor(self.rewards[idx], device=self.device),
            "next_state": torch.as_tensor(self.next_states[idx], device=self.device),
            "done": torch.as_tensor(self.dones[idx], device=self.device),
        }

    def __len__(self) -> int:
        return self.size
