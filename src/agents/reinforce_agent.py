import numpy as np
from torch import Tensor

from src.utils.config import Config
from src.agents.baselines import BaseBaseline


class ReinforceAgent:
    """
    REINFORCE (Williams 1992) policy gradient agent.
    Collects a full episode, then updates the policy using
    Monte Carlo returns G_t with a subtracted baseline b(s_t):

        grad J = E[ sum_t grad log pi(a_t|s_t) * (G_t - b(s_t)) ]

    The baseline is pluggable: any object implementing BaseBaseline.
    Default: ZeroBaseline (vanilla REINFORCE, b=0).
    """

    def __init__(self, config: Config,
                 baseline: 'BaseBaseline | None' = None) -> None:
        """
        Args:
            config:   Config with all hyperparameters.
            baseline: BaseBaseline instance. Defaults to ZeroBaseline().
        """
        pass

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """
        Sample action from pi(a|s).

        Args:
            state:         current state vector.
            deterministic: use mean action if True (for evaluation).

        Returns:
            action: numpy array [delta_theta, f].
        """
        pass

    def store_transition(self, state: np.ndarray,
                         action: np.ndarray,
                         reward: float,
                         log_prob: Tensor) -> None:
        """
        Store one transition from the current episode.
        Must be called after every env.step() during collection.

        Args:
            state:    state s_t.
            action:   action a_t.
            reward:   reward r_t = r(s_t, a_t).
            log_prob: log pi(a_t | s_t) computed by the actor.
        """
        pass

    def update(self) -> dict[str, float]:
        """
        Policy gradient update at the end of an episode.

        Steps:
            1. Compute returns: G_t = sum_{k>=t} gamma^(k-t) * r_k.
            2. Query baseline: b_t = baseline.estimate(state_t).
            3. Compute advantages: A_t = G_t - b_t.
            4. Policy loss: L = -sum_t log_prob_t * A_t.
            5. Backpropagate and update actor weights.
            6. Update baseline: baseline.update(states, returns).
            7. Clear episode buffer.

        Returns:
            metrics: dict with policy_loss, mean_return, mean_advantage.
        """
        pass

    def save(self, path: str) -> None:
        """Save actor weights and baseline state to disk."""
        pass

    def load(self, path: str) -> None:
        """Load actor weights and baseline state from disk."""
        pass
