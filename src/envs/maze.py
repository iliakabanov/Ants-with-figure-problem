from __future__ import annotations

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
        Build room boundaries and interior wall slabs in the pymunk space.
        Each vertical wall is two axis-aligned polygons (below / above the gap).

        Args:
            config: Config with room dimensions, gap_width, wall x-positions.
            space:  pymunk Space to add static segments to.
        """
        self._cfg = config
        self._space = space
        self._static = space.static_body

        rw = config.room_width
        rh = config.room_height
        side = (rw - config.corridor_length) * 0.5
        self._x_walls = [side, side + config.corridor_length]

        self._radius = 0.5
        # Solid thickness of outer frame (world units). Segments with radius gave weak corners.
        self._border_thickness = max(self._radius * 2.0, 1.0)
        self._border_polys: list[pymunk.Poly] = []
        # Interior walls as axis-aligned Polys (half-width thickness/2). Segment+radius bulges into the gap.
        self._wall_polys: list[tuple[pymunk.Poly | None, pymunk.Poly | None]] = []

        self._y_gaps = [rh * 0.5, rh * 0.5]

        self._add_room_bounds(rw, rh)
        for _ in self._x_walls:
            self._wall_polys.append((None, None))
        self._sync_wall_segments()

    def _add_room_bounds(self, rw: float, rh: float) -> None:
        """
        Outer walls as filled rectangles **outside** [0, rw] × [0, rh].
        This avoids corner gaps and inconsistent normals vs thin Segments + radius.
        """
        th = self._border_thickness
        static = self._static
        # Bottom strip (includes lower-left / lower-right corner blocks)
        bottom = pymunk.Poly(
            static,
            [(-th, -th), (rw + th, -th), (rw + th, 0.0), (-th, 0.0)],
        )
        # Top strip
        top = pymunk.Poly(
            static,
            [(-th, rh), (rw + th, rh), (rw + th, rh + th), (-th, rh + th)],
        )
        # Left strip (between bottom and top slabs; overlaps corner regions — OK)
        left = pymunk.Poly(
            static,
            [(-th, 0.0), (0.0, 0.0), (0.0, rh), (-th, rh)],
        )
        # Right strip
        right = pymunk.Poly(
            static,
            [(rw, 0.0), (rw + th, 0.0), (rw + th, rh), (rw, rh)],
        )
        for poly in (bottom, top, left, right):
            poly.friction = 0.7
            poly.elasticity = 0.0
            self._border_polys.append(poly)
            self._space.add(poly)

    def _sync_wall_segments(self) -> None:
        rh = self._cfg.room_height
        half_gap = self._cfg.gap_width * 0.5
        hw = self._cfg.thickness * 0.5

        for i, (xw, yc) in enumerate(zip(self._x_walls, self._y_gaps)):
            old_lower, old_upper = self._wall_polys[i]
            for old in (old_lower, old_upper):
                if old is not None:
                    self._space.remove(old)

            y_lo = max(0.0, yc - half_gap)
            y_hi = min(rh, yc + half_gap)
            x_left = xw - hw
            x_right = xw + hw

            new_lower: pymunk.Poly | None = None
            if y_lo > 1e-6:
                new_lower = pymunk.Poly(
                    self._static,
                    [
                        (x_left, 0.0),
                        (x_right, 0.0),
                        (x_right, y_lo),
                        (x_left, y_lo),
                    ],
                )
                new_lower.friction = 0.7
                new_lower.elasticity = 0.0
                self._space.add(new_lower)

            new_upper: pymunk.Poly | None = None
            if y_hi < rh - 1e-6:
                new_upper = pymunk.Poly(
                    self._static,
                    [
                        (x_left, y_hi),
                        (x_right, y_hi),
                        (x_right, rh),
                        (x_left, rh),
                    ],
                )
                new_upper.friction = 0.7
                new_upper.elasticity = 0.0
                self._space.add(new_upper)

            self._wall_polys[i] = (new_lower, new_upper)

        self._space.reindex_static()

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
        g = self._cfg.gap_width
        return [
            {'x': self._x_walls[0], 'y_gap': self._y_gaps[0], 'gap_width': g},
            {'x': self._x_walls[1], 'y_gap': self._y_gaps[1], 'gap_width': g},
        ]

    def randomise_gaps(self, rng: np.random.Generator) -> None:
        """
        Randomly reposition the gap centre y_gap_i for each wall pair.
        Called by MazeEnv.reset() to generate a new labyrinth layout.
        Gap centres are sampled uniformly within the room, leaving
        enough margin so the gap does not touch the floor or ceiling.

        Args:
            rng: numpy random generator (seeded for reproducibility).
        """
        rh = self._cfg.room_height
        low = self._cfg.gap_margin
        high = rh - self._cfg.gap_margin
        if self._cfg.randomise_gaps:
            self._y_gaps[0] = float(rng.uniform(low, high))
            self._y_gaps[1] = float(rng.uniform(low, high))
        else:
            mid = rh * 0.5
            self._y_gaps[0] = mid
            self._y_gaps[1] = mid
        self._sync_wall_segments()

    def is_out_of_bounds(self,
                         corners: list[tuple[float, float]]) -> bool:
        """Check predicate B(s): any corner outside room boundary."""
        rw, rh = self._cfg.room_width, self._cfg.room_height
        eps = 1e-6
        for x, y in corners:
            if x < -eps or x > rw + eps or y < -eps or y > rh + eps:
                return True
        return False
