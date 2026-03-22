from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pymunk


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    n = len(vertices)
    s = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _clip_polygon_x_ge(vertices: list[tuple[float, float]], x_wall: float) -> list[tuple[float, float]]:
    """Clip a closed polygon (CCW) to the half-plane x >= x_wall."""

    def inside(p: tuple[float, float]) -> bool:
        return p[0] >= x_wall

    def intersect(s: tuple[float, float], e: tuple[float, float]) -> tuple[float, float]:
        sx, sy = s
        ex, ey = e
        if abs(ex - sx) < 1e-12:
            return (x_wall, sy)
        t = (x_wall - sx) / (ex - sx)
        return (x_wall, sy + t * (ey - sy))

    out: list[tuple[float, float]] = []
    if not vertices:
        return out
    prev = vertices[-1]
    for cur in vertices:
        cin = inside(cur)
        pin = inside(prev)
        if cin:
            if not pin:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif pin:
            out.append(intersect(prev, cur))
        prev = cur
    return out


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
    wall_x = float(wall_geometry["x"])
    clipped = _clip_polygon_x_ge(polygon, wall_x)
    return _polygon_area(clipped)


def compass8_dir_to_body_rad(direction_index: int) -> float:
    """8 направлений с шагом 45° в системе тела: 0 = север (+y), далее по часовой."""
    i = int(direction_index) % 8
    return 0.5 * math.pi - i * (0.25 * math.pi)


def cast_rays_detailed(
    origins: list[tuple[float, float]],
    directions: list[float],
    space: pymunk.Space,
    r_max: float,
    ignore_bodies: Iterable[pymunk.Body] | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[bool]]:
    """
    Same physics as :func:`cast_rays`, but also returns ray endpoints and hit flags.

    Returns:
        distances: shape (N,) with N = len(origins)*len(directions).

        endpoints: same length; each point ``(x, y)`` lies on the ray at the
        reported distance (collision point or max range).

        hit_obstacle: ``True`` if the ray hit geometry before ``r_max``,
        ``False`` if it reached the sensing range without a hit.
    """
    ignore = frozenset(ignore_bodies) if ignore_bodies is not None else frozenset()
    filt = pymunk.ShapeFilter()
    eps = 1e-2
    out_d: list[float] = []
    out_p: list[tuple[float, float]] = []
    out_hit: list[bool] = []

    for ox, oy in origins:
        for direction in directions:
            c = math.cos(direction)
            s = math.sin(direction)
            sx = ox + c * eps
            sy = oy + s * eps
            ex = ox + c * r_max
            ey = oy + s * r_max

            hits = space.segment_query((sx, sy), (ex, ey), 0.0, filt)
            best: float | None = None
            for hit in hits:
                if hit.shape.body in ignore:
                    continue
                d = math.hypot(hit.point.x - ox, hit.point.y - oy)
                if d < 1e-6:
                    continue
                if best is None or d < best:
                    best = d

            if best is None:
                dist = r_max
                hit = False
            else:
                dist = min(best, r_max)
                hit = True

            px = ox + c * dist
            py = oy + s * dist
            out_d.append(dist)
            out_p.append((px, py))
            out_hit.append(hit)

    return (
        np.asarray(out_d, dtype=np.float64),
        out_p,
        out_hit,
    )


def cast_rays_detailed_paired(
    origins: list[tuple[float, float]],
    directions: list[float],
    space: pymunk.Space,
    r_max: float,
    ignore_bodies: Iterable[pymunk.Body] | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[bool]]:
    """Как :func:`cast_rays_detailed`, но луч ``i`` = ``(origins[i], directions[i])`` (попарно)."""
    if len(origins) != len(directions):
        raise ValueError("origins and directions must have the same length")
    ignore = frozenset(ignore_bodies) if ignore_bodies is not None else frozenset()
    filt = pymunk.ShapeFilter()
    eps = 1e-2
    out_d: list[float] = []
    out_p: list[tuple[float, float]] = []
    out_hit: list[bool] = []

    for (ox, oy), direction in zip(origins, directions):
        c = math.cos(direction)
        s = math.sin(direction)
        sx = ox + c * eps
        sy = oy + s * eps
        ex = ox + c * r_max
        ey = oy + s * r_max

        hits = space.segment_query((sx, sy), (ex, ey), 0.0, filt)
        best: float | None = None
        for hit in hits:
            if hit.shape.body in ignore:
                continue
            d = math.hypot(hit.point.x - ox, hit.point.y - oy)
            if d < 1e-6:
                continue
            if best is None or d < best:
                best = d

        if best is None:
            dist = r_max
            hit = False
        else:
            dist = min(best, r_max)
            hit = True

        px = ox + c * dist
        py = oy + s * dist
        out_d.append(dist)
        out_p.append((px, py))
        out_hit.append(hit)

    return (
        np.asarray(out_d, dtype=np.float64),
        out_p,
        out_hit,
    )


def cast_rays(origins: list[tuple[float, float]],
              directions: list[float],
              space: pymunk.Space,
              r_max: float,
              ignore_bodies: Iterable[pymunk.Body] | None = None) -> np.ndarray:
    """
    Cast rays from given origins in given directions.
    Returns distances to nearest obstacle, clipped to r_max.

    Uses pymunk segment queries against static and dynamic geometry.
    Shapes whose bodies are listed in ``ignore_bodies`` are skipped
    (e.g. the controlled figure so corners do not immediately hit themselves).

    Args:
        origins:    ray start points (figure corners).
        directions: ray angles in radians (world frame).
        space:      pymunk space with all obstacles.
        r_max:      maximum sensing range.
        ignore_bodies: bodies to treat as transparent (optional).

    Returns:
        distances: array of shape (len(origins)*len(directions),),
                   order: for each origin, all directions in order.
    """
    d, _, _ = cast_rays_detailed(origins, directions, space, r_max, ignore_bodies)
    return d
