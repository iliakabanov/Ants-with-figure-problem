import numpy as np
import pymunk


def cast_rays(origins: list[tuple[float, float]],
              directions: list[float],
              space: pymunk.Space,
              r_max: float) -> np.ndarray:
    """
    Cast rays from given origins in given directions.
    Returns distances to nearest obstacle, clipped to r_max.

    Args:
        origins:    ray start points (figure corners).
        directions: ray angles in radians.
        space:      pymunk space with all obstacles.
        r_max:      maximum sensing range.

    Returns:
        distances: array of shape (len(origins)*len(directions),).
    """
    pass


def compute_area_past_wall(polygon: list[tuple[float, float]],
                           wall_geometry: dict) -> float:
    """
    Compute area of a polygon lying to the right of wall_geometry['x'].
    wall_geometry: { 'x': float, 'y_gap': float, 'gap_width': float }
    Used to evaluate progress variable rho_i(s).

    Args:
        polygon:       list of (x, y) vertices in world space.
        wall_geometry: dict describing the wall pair.

    Returns:
        area: float >= 0.
    """
    pass
