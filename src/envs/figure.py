from __future__ import annotations

import pymunk

from src.utils.config import Config
from src.utils.geometry import compute_area_past_wall


def _moment_t_figure(masses: tuple[float, float, float],
                     sizes: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
                     offsets_y: tuple[float, float, float]) -> float:
    """Sum of box moments about the origin (COM) along y for symmetric T in x."""
    mt, ml, mb = masses
    (wt, ht), (wl, hl), (wb, hb) = sizes
    yt, yl, yb = offsets_y
    return (
        pymunk.moment_for_box(mt, (wt, ht)) + mt * yt * yt
        + pymunk.moment_for_box(ml, (wl, hl)) + ml * yl * yl
        + pymunk.moment_for_box(mb, (wb, hb)) + mb * yb * yb
    )


def _outline_local(top_len: float, leg_len: float, bottom_len: float, t: float) -> list[tuple[float, float]]:
    """
    CCW outer boundary of the T in body frame with COM at (0, 0).
    Layout: top bar on top (length top_len), leg down (length leg_len, thickness t),
    bottom bar (length bottom_len) under the leg.
    """
    area = top_len * t + leg_len * t + bottom_len * t
    y_com = (top_len * t * (t * 0.5) + leg_len * t * (-leg_len * 0.5) + bottom_len * t * (-leg_len - t * 0.5)) / area
    top_cy = t * 0.5 - y_com
    leg_cy = -leg_len * 0.5 - y_com
    bot_cy = -leg_len - t * 0.5 - y_com
    ht = t * 0.5
    y_top_hi = top_cy + ht
    y_top_lo = top_cy - ht
    y_leg_bot = leg_cy - leg_len * 0.5
    y_bot_lo = bot_cy - ht
    hx_top = top_len * 0.5
    hx_leg = t * 0.5
    hx_bot = bottom_len * 0.5
    return [
        (-hx_top, y_top_hi),
        (hx_top, y_top_hi),
        (hx_top, y_top_lo),
        (hx_leg, y_top_lo),
        (hx_leg, y_leg_bot),
        (hx_bot, y_leg_bot),
        (hx_bot, y_bot_lo),
        (-hx_bot, y_bot_lo),
        (-hx_bot, y_leg_bot),
        (-hx_leg, y_leg_bot),
        (-hx_leg, y_top_lo),
        (-hx_top, y_top_lo),
    ]


class TFigure:
    """
    Rigid T-shaped figure: top bar (5x1), leg (4x1), bottom bar (2x1).
    Wraps a pymunk Body with three rectangular shapes.
    """

    def __init__(self, space: pymunk.Space, config: Config | None = None) -> None:
        cfg = config or Config()
        self._cfg = cfg
        tl, ll, bl, t = cfg.top_bar_length, cfg.leg_length, cfg.bottom_bar_length, cfg.thickness
        self._outline_local = _outline_local(tl, ll, bl, t)

        area_total = tl * t + ll * t + bl * t
        mt = tl * t
        ml = ll * t
        mb = bl * t
        area = area_total
        y_com = (mt * (t * 0.5) + ml * (-ll * 0.5) + mb * (-ll - t * 0.5)) / area
        top_cy = t * 0.5 - y_com
        leg_cy = -ll * 0.5 - y_com
        bot_cy = -ll - t * 0.5 - y_com

        moment = _moment_t_figure(
            (mt, ml, mb),
            ((tl, t), (t, ll), (bl, t)),
            (top_cy, leg_cy, bot_cy),
        )
        self._body = pymunk.Body(area_total, moment)
        self._shapes: list[pymunk.Shape] = []

        def box_verts(cx: float, cy: float, half_w: float, half_h: float) -> list[tuple[float, float]]:
            return [
                (cx - half_w, cy - half_h),
                (cx + half_w, cy - half_h),
                (cx + half_w, cy + half_h),
                (cx - half_w, cy + half_h),
            ]

        parts = (
            (tl * 0.5, t * 0.5, top_cy),
            (t * 0.5, ll * 0.5, leg_cy),
            (bl * 0.5, t * 0.5, bot_cy),
        )
        for hw, hh, cy in parts:
            verts = box_verts(0.0, cy, hw, hh)
            sh = pymunk.Poly(self._body, verts)
            sh.friction = 0.7
            sh.elasticity = 0.05
            self._shapes.append(sh)
        space.add(self._body, *self._shapes)
        self._total_area = float(area_total)

    @property
    def body(self) -> pymunk.Body:
        return self._body

    def set_state(self, x: float, y: float, theta: float) -> None:
        """Set figure position (centre of mass) and orientation (radians)."""
        self._body.position = (x, y)
        self._body.angle = float(theta)
        self._body.velocity = (0.0, 0.0)
        self._body.angular_velocity = 0.0

    def _to_world(self, lx: float, ly: float) -> tuple[float, float]:
        v = pymunk.Vec2d(lx, ly).rotated(self._body.angle) + self._body.position
        return (float(v.x), float(v.y))

    def get_corners(self) -> list[tuple[float, float]]:
        """Return world-space coordinates of all outer corners of the T outline."""
        return [self._to_world(px, py) for px, py in self._outline_local]

    def compute_progress(self, wall_geometry: dict) -> float:
        """
        rho_i(s) = S_i_wall(s) / S_total for a wall pair:
        only the part of the figure with x >= wall_geometry['x'] counts.
        Keys 'y_gap' and 'gap_width' are ignored (progress is purely by wall x).
        """
        area_past = compute_area_past_wall(self.get_corners(), wall_geometry)
        return max(0.0, min(1.0, area_past / self._total_area))

    def get_total_area(self) -> float:
        """Return total area S_total of the T-figure."""
        return self._total_area
