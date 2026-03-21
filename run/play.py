import numpy as np

from src.utils.config import Config


def run_interactive(config: Config) -> None:
    """
    Main interactive loop.
    Each pygame frame:
        1. Read keyboard state.
        2. Map keys to action (delta_theta, f).
        3. Call env.step(action).
        4. Render updated state.
        5. Display HUD: step count, reward, rho_1, rho_2.

    Args:
        config: Config with environment hyperparameters.
    """
    pass


def keys_to_action(keys: dict, config: Config) -> np.ndarray:
    """
    Convert pressed keys to a continuous action vector.
    Simultaneous key presses are supported:
        e.g. RIGHT + UP rotates and pushes at the same time.

    Args:
        keys:   dict of pygame key states (from pygame.key.get_pressed()).
        config: Config for max_delta_theta and max_thrust values.

    Returns:
        action: numpy array [delta_theta, f].
                Returns [0.0, 0.0] if no relevant key is pressed.
    """
    pass


def render_hud(surface, step: int,
               total_reward: float,
               rho_1: float, rho_2: float) -> None:
    """
    Draw heads-up display overlay on the pygame surface.
    Shows: current step, cumulative reward, progress rho_1 and rho_2
    as text in the top-left corner of the window.

    Args:
        surface:      pygame surface to draw on.
        step:         current step count in the episode.
        total_reward: cumulative reward so far.
        rho_1:        progress past wall pair 1 in [0, 1].
        rho_2:        progress past wall pair 2 in [0, 1].
    """
    pass


if __name__ == "__main__":
    """
    Play the labyrinth game manually using the keyboard.
    The environment runs in real time with pygame rendering.

    Controls:
        LEFT  arrow — rotate figure counter-clockwise (delta_theta = -max)
        RIGHT arrow — rotate figure clockwise          (delta_theta = +max)
        UP    arrow — push figure forward              (f = +max)
        DOWN  arrow — push figure backward             (f = -max)
        R           — reset episode
        Q / ESC     — quit

    Usage: python run/play.py
    """
    pass
