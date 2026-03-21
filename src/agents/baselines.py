import numpy as np
from abc import ABC, abstractmethod


class BaseBaseline(ABC):
    """
    Abstract base class for REINFORCE baselines.
    Subclass this to add new baseline variants without
    modifying ReinforceAgent or any other file.
    """

    @abstractmethod
    def estimate(self, state: np.ndarray) -> float:
        """
        Return baseline value b(s) for the given state.

        Args:
            state: state vector s_t.

        Returns:
            b: scalar float.
        """
        pass

    @abstractmethod
    def update(self, states: list[np.ndarray],
               returns: list[float]) -> None:
        """
        Update baseline using (state, return) pairs from the episode.

        Args:
            states:  list of state vectors from the completed episode.
            returns: list of Monte Carlo returns G_t.
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serialisable state for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore baseline state from a checkpoint dict."""
        pass


# ── Concrete implementations ──────────────────────────────────

class ZeroBaseline(BaseBaseline):
    """
    Trivial baseline: b(s) = 0 for all s.
    Equivalent to vanilla REINFORCE with no variance reduction.
    Default starting point — replace with a better baseline later.
    """

    def estimate(self, state: np.ndarray) -> float:
        """Always returns 0.0."""
        pass

    def update(self, states: list[np.ndarray],
               returns: list[float]) -> None:
        """No-op: nothing to update."""
        pass

    def state_dict(self) -> dict:
        pass

    def load_state_dict(self, state: dict) -> None:
        pass


class MeanReturnBaseline(BaseBaseline):
    """
    Running mean baseline: b(s) = exponential moving average of G_t.
    Reduces variance by centering returns around their running mean.
    Updated once per episode.
    """

    def __init__(self, momentum: float = 0.99) -> None:
        """
        Args:
            momentum: EMA decay factor (closer to 1 = slower update).
        """
        pass

    def estimate(self, state: np.ndarray) -> float:
        """Return current running mean of returns (ignores state)."""
        pass

    def update(self, states: list[np.ndarray],
               returns: list[float]) -> None:
        """Update running mean with mean return of the episode."""
        pass

    def state_dict(self) -> dict:
        pass

    def load_state_dict(self, state: dict) -> None:
        pass
