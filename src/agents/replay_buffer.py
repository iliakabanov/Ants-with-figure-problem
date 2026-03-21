import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Circular buffer storing (s, a, r, s', done). Used by Actor-Critic only."""

    def __init__(self, capacity: int, state_dim: int,
                 action_dim: int) -> None:
        pass

    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray,
             done: bool) -> None:
        """Store one transition."""
        pass

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        """Sample random minibatch. Keys: state, action, reward, next_state, done."""
        pass

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        pass
