"""
Baseline abstraction and implementations for the ReinforceAgent.

A baseline b(s) is subtracted from the Monte Carlo return G_t to reduce
the variance of the policy gradient estimate:

    grad J(theta) = E[ sum_t grad log pi(a_t|s_t) * (G_t - b(s_t)) ]

Subtracting b(s) does not introduce bias as long as b does not depend
on the action a_t. It only reduces variance.

To add a new baseline, subclass BaseBaseline and implement the three
abstract methods: estimate(), update(), state_dict(), load_state_dict().
No other file needs to be modified.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseBaseline(ABC):
    """
    Abstract base class for REINFORCE baselines.

    All baselines must implement estimate() and update().
    The state_dict() / load_state_dict() pair is used for checkpointing.
    """

    @abstractmethod
    def estimate(self, state: np.ndarray) -> float:
        """
        Return the baseline value b(s) for the given state.

        This value will be subtracted from the Monte Carlo return G_t
        to form the advantage estimate A_t = G_t - b(s_t).

        Args:
            state: state vector s_t, shape (state_dim,).

        Returns:
            b: scalar float baseline value.
        """

    @abstractmethod
    def update(
        self,
        states: list[np.ndarray],
        returns: list[float],
    ) -> None:
        """
        Update the baseline using observed (state, return) pairs from
        the completed episode.

        Called once per episode after all returns G_t have been computed,
        before the episode buffer is cleared.

        Args:
            states:  list of state vectors s_t from the episode,
                     length = episode length.
            returns: list of Monte Carlo returns G_t for each step,
                     length = episode length.
        """

    @abstractmethod
    def state_dict(self) -> dict:
        """
        Return a serialisable dict of the baseline's internal state.
        Used for saving checkpoints via agent.save().
        """

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """
        Restore the baseline's internal state from a checkpoint dict.
        Used for loading checkpoints via agent.load().

        Args:
            state: dict previously returned by state_dict().
        """


class ZeroBaseline(BaseBaseline):
    """
    Trivial baseline: b(s) = 0 for all s.

    Equivalent to vanilla REINFORCE with no variance reduction.
    The policy gradient update reduces to:

        grad J(theta) = E[ sum_t grad log pi(a_t|s_t) * G_t ]

    This is the default starting point. Replace with a better baseline
    (e.g. MeanReturnBaseline) to reduce gradient variance and speed up
    learning.
    """

    def estimate(self, state: np.ndarray) -> float:
        """
        Always returns 0.0 regardless of the state.

        Args:
            state: state vector (ignored).

        Returns:
            0.0
        """
        return 0.0

    def update(
        self,
        states: list[np.ndarray],
        returns: list[float],
    ) -> None:
        """
        No-op: ZeroBaseline has no internal state to update.

        Args:
            states:  list of state vectors (ignored).
            returns: list of returns (ignored).
        """

    def state_dict(self) -> dict:
        """
        Returns an empty dict — ZeroBaseline has no internal state.
        """
        return {}

    def load_state_dict(self, state: dict) -> None:
        """
        No-op: nothing to restore.

        Args:
            state: dict (ignored).
        """
