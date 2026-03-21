import numpy as np

from src.utils.config import Config


class ActorCriticAgent:
    """
    Actor-Critic agent for continuous action spaces.
    Combines GaussianActor, Critic, and ReplayBuffer.
    """

    def __init__(self, config: Config) -> None:
        """Initialise actor, critics, target networks, optimisers, buffer."""
        pass

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """Sample action from pi(a|s). Deterministic=True for eval."""
        pass

    def update(self) -> dict[str, float]:
        """
        Sample minibatch and perform one gradient update.

        Returns:
            dict with critic_loss, actor_loss, alpha_loss.
        """
        pass

    def save(self, path: str) -> None:
        """Save all network weights and optimiser states."""
        pass

    def load(self, path: str) -> None:
        """Load network weights and optimiser states."""
        pass
