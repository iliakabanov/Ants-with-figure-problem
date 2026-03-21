import numpy as np
import pymunk

from src.utils.config import Config


class Maze:
    """
    Rectangular labyrinth with two pairs of vertical walls.
    Each wall pair i is fully described by:
      - x_wall_i  : x-coordinate of the wall pair
      - y_gap_i   : y-coordinate of the gap centre (variable per pair)
      - gap_width : width of the gap (fixed, from Config)
    The gap position y_gap_i can be randomised each episode
    to create a non-deterministic labyrinth.
    """

    def __init__(self, config: Config, space: pymunk.Space) -> None:
        """
        Build room boundaries and wall segments in the pymunk space.
        Wall segments are split into top and bottom parts around the gap.

        Args:
            config: Config with room dimensions, gap_width, wall x-positions.
            space:  pymunk Space to add static segments to.
        """
        pass

    def get_wall_geometries(self) -> list[dict]:
        """
        Return full geometry of all wall pairs.
        Each entry is a dict:
            {
                'x':         float,  # x-coordinate of this wall pair
                'y_gap':     float,  # y-coordinate of gap centre
                'gap_width': float   # gap width (same for all pairs)
            }
        Used by TFigure.compute_progress() and _compute_reward().

        Returns:
            list of length 2, one dict per wall pair.
        """
        pass

    def randomise_gaps(self, rng: np.random.Generator) -> None:
        """
        Randomly reposition the gap centre y_gap_i for each wall pair.
        Called by MazeEnv.reset() to generate a new labyrinth layout.
        Gap centres are sampled uniformly within the room, leaving
        enough margin so the gap does not touch the floor or ceiling.

        Args:
            rng: numpy random generator (seeded for reproducibility).
        """
        pass

    def is_out_of_bounds(self,
                         corners: list[tuple[float, float]]) -> bool:
        """Check predicate B(s): any corner outside room boundary."""
        pass
