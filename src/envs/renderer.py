import numpy as np

from src.utils.config import Config


class Renderer:
    """Renders maze and figure via pygame or matplotlib."""

    def __init__(self, config: Config) -> None:
        """Initialise renderer with window size and colour scheme."""
        pass

    def render(self, env_state: dict) -> np.ndarray | None:
        """Draw current state. Returns RGB array or None."""
        pass

    def close(self) -> None:
        """Destroy pygame window and free resources."""
        pass
