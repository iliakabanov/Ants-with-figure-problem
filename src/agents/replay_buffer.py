"""
Replay buffer for the ActorCriticAgent.

Follows the lecture definition exactly:
    R_t := { S_{t-k-1}, A_{t-k-1}, S_{t-k}, A_{t-k} }_{k=0:M}

The buffer stores the last M+1 transitions in order.
When get_all() is called, all M+1 transitions are returned as a batch —
no random sampling. This is the on-policy variant from the lecture.

Once the buffer holds M+1 transitions, each new push() overwrites
the oldest one (circular buffer of size M+1).
"""

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """
    Circular buffer storing the last M+1 environment transitions.

    Unlike the standard large random-sample replay buffer, this buffer
    matches the lecture formulation: the critic loss is computed over
    all M+1 most recent transitions, not a random subset.

    The buffer size M+1 is small (e.g. 64), so all transitions are
    recent and collected by the current policy. This keeps the training
    closer to the on-policy setting assumed in the lecture's convergence
    analysis.

    Args:
        M:          number of past transitions to keep (buffer holds M+1).
        state_dim:  dimensionality of the state vector.
        action_dim: dimensionality of the action vector.
        device:     torch device for returned tensors.
    """

    def __init__(
        self,
        M: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = M + 1  # buffer holds exactly M+1 transitions
        self.device = device
        self.ptr = 0           # pointer to the next write position
        self.size = 0          # number of transitions currently stored

        # Pre-allocate numpy arrays for efficiency
        self.states      = np.zeros((self.capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((self.capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((self.capacity, 1),          dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition, overwriting the oldest if buffer is full.

        Args:
            state:      state vector s_t,          shape (state_dim,).
            action:     action vector a_t,          shape (action_dim,).
            reward:     scalar reward r(s_t, a_t).
            next_state: next state vector s_{t+1},  shape (state_dim,).
            done:       True if the episode ended after this transition.
        """
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_all(self) -> dict[str, Tensor]:
        """
        Return all stored transitions as a single batch.

        This corresponds to the sum over k=0..M in the lecture loss:
            L_crit(omega | R_t) = 0.5 * sum_{k=0}^{M} (delta(d_{t-k}))^2

        All currently stored transitions are returned in insertion order.
        The batch size equals self.size (up to M+1).

        Returns:
            batch: dict with keys 'state', 'action', 'reward',
                   'next_state', 'done' — all torch tensors on self.device.
        """
        # Reconstruct insertion order from circular buffer
        if self.size < self.capacity:
            # Buffer not yet full — take first self.size entries
            indices = np.arange(self.size)
        else:
            # Buffer full — oldest entry is at self.ptr
            indices = np.arange(self.ptr, self.ptr + self.capacity) % self.capacity

        return {
            "state":      torch.FloatTensor(self.states[indices]).to(self.device),
            "action":     torch.FloatTensor(self.actions[indices]).to(self.device),
            "reward":     torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_state": torch.FloatTensor(self.next_states[indices]).to(self.device),
            "done":       torch.FloatTensor(self.dones[indices]).to(self.device),
        }

    def is_ready(self) -> bool:
        """
        Return True if the buffer holds at least one transition.

        The critic update can be called as soon as the first transition
        is stored — no warmup period needed, unlike the large random buffer.
        """
        return self.size > 0

    def __len__(self) -> int:
        """Return the number of transitions currently stored."""
        return self.size
