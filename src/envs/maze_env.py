import numpy as np
import gymnasium

from src.utils.config import Config


class MazeEnv(gymnasium.Env):
    """
    Gymnasium environment for the figure-navigation task.
    The agent controls a rigid T-shaped figure and must guide it
    through two pairs of narrow wall gaps from start to finish.

    State: (x, y, theta, d_1, ..., d_K)
      - x, y  : centre-of-mass coordinates
      - theta  : rotation angle in radians
      - d_i    : ray distances from figure corners to nearest obstacle

    Action: (delta_theta, f)
      - delta_theta : rotation increment in [-5 deg, 5 deg]
      - f           : thrust force in [-0.1, 0.1]
    """

    def __init__(self, config: Config) -> None:
        """
        Initialise the environment.

        Args:
            config: Config dataclass with all hyperparameters.
        """
        pass

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to the initial state.

        Args:
            seed: optional random seed for reproducibility.

        Returns:
            state: initial state vector.
            info:  auxiliary diagnostic information.
        """
        pass

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action and advance environment by one step.
        Computes candidate next state, checks C(s_tilde) and B(s_tilde),
        applies rollback if needed, then computes reward r(s, a).

        Args:
            action: array [delta_theta, f].

        Returns:
            state:      next state vector.
            reward:     scalar reward r(s, a).
            terminated: True if finish reached.
            truncated:  True if max steps exceeded.
            info:       dict with rho_1, rho_2, collision flags.
        """
        pass

    def render(self) -> np.ndarray | None:
        """Return RGB array if render_mode=rgb_array, else None."""
        pass

    def seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        Seeds both numpy and the internal random generator used
        by Maze.randomise_gaps().

        Args:
            seed: integer seed value.
        """
        pass

    def close(self) -> None:
        """Release rendering resources."""
        pass

    def _compute_state(self) -> np.ndarray:
        """
        Build the full state vector (x, y, theta, d_1, ..., d_K).
        Casts rays from all figure corners and clips distances to R_max.

        Returns:
            state: flat numpy array of shape (3 + K,).
        """
        pass

    def _compute_reward(self, s, s_next,
                        collision: bool, out_of_bounds: bool) -> float:
        """
        Compute reward r(s, a) = r_fin*1_fin - r_col*1_col
        - r_oob*1_oob + r_wall*(delta_rho_1 + delta_rho_2) - r_step.

        Args:
            s:             state before action.
            s_next:        state after action (post-rollback).
            collision:     whether C(s_tilde) was true.
            out_of_bounds: whether B(s_tilde) was true.

        Returns:
            reward: scalar float.
        """
        pass
