from __future__ import annotations

import numpy as np
import pygame

from src.utils.config import Config


class Renderer:
    """Renders maze and figure with pygame (window or off-screen surface)."""

    def __init__(self, config: Config, render_mode: str = "rgb_array") -> None:
        """
        Initialise renderer with window size and colour scheme.

        Args:
            config: room dimensions from config.
            render_mode: 'human' opens a window; 'rgb_array' draws off-screen only.
        """
        self._cfg = config
        self._mode = render_mode
        self._scale = 28.0
        self._pad = 16
        rw = config.room_width
        rh = config.room_height
        self._surf_w = int(self._pad * 2 + rw * self._scale)
        self._surf_h = int(self._pad * 2 + rh * self._scale)

        self._bg = (245, 245, 240)
        self._wall = (55, 55, 70)
        self._border = (30, 30, 40)
        self._figure_fill = (200, 90, 70)
        self._figure_edge = (120, 50, 40)
        self._ray_line = (80, 150, 110)
        self._ray_hit = (235, 100, 35)
        self._ray_open = (100, 130, 190)

        self._wall_line_px = max(2, int(self._scale * 1.0))
        self._border_px = max(2, int(self._scale * 0.15))

        self._screen: pygame.Surface | None = None
        self._canvas = pygame.Surface((self._surf_w, self._surf_h))

        pygame.init()
        pygame.display.set_caption("Figure maze")

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = self._pad + x * self._scale
        sy = self._surf_h - self._pad - y * self._scale
        return int(round(sx)), int(round(sy))

    def _draw_rays(self, surface: pygame.Surface, env_state: dict) -> None:
        """Draw ray segments and hit / max-range markers when ``env_state`` includes ray data."""
        origins = env_state.get("ray_origins")
        endpts = env_state.get("ray_endpoints")
        hits = env_state.get("ray_hits")
        if not origins or not endpts or not hits:
            return
        n = min(len(origins), len(endpts), len(hits))
        for i in range(n):
            p0 = self._world_to_screen(float(origins[i][0]), float(origins[i][1]))
            p1 = self._world_to_screen(float(endpts[i][0]), float(endpts[i][1]))
            pygame.draw.line(surface, self._ray_line, p0, p1, 1)
            if hits[i]:
                pygame.draw.circle(surface, self._ray_hit, p1, 4, 0)
            else:
                pygame.draw.circle(surface, self._ray_open, p1, 3, 0)

    @property
    def canvas(self) -> pygame.Surface:
        """Back buffer; safe to draw HUD on after :meth:`draw_world`."""
        return self._canvas

    def draw_world(self, env_state: dict) -> None:
        """
        Draw maze and figure onto the internal canvas only (no display update).

        env_state keys used:
            room_width, room_height — fall back to Config if missing.
            wall_geometries — list of dicts with x, y_gap, gap_width (Maze format).
            figure_corners — list of (x, y) in world space (CCW outline).

        Optional ray overlay (same length, order as state rays):
            ray_origins — one start per ray (flattened corners × directions);
            ray_endpoints — (x, y) end of each ray;
            ray_hits — True if obstacle hit before r_max, else open range.
        """
        self._canvas.fill(self._bg)

        rw = float(env_state.get("room_width", self._cfg.room_width))
        rh = float(env_state.get("room_height", self._cfg.room_height))
        walls = env_state.get("wall_geometries", [])
        corners = env_state.get("figure_corners", [])

        bl = self._world_to_screen(0.0, 0.0)
        br = self._world_to_screen(rw, 0.0)
        tr = self._world_to_screen(rw, rh)
        tl = self._world_to_screen(0.0, rh)
        pygame.draw.lines(self._canvas, self._border, True, [bl, br, tr, tl], self._border_px)

        for wg in walls:
            xw = float(wg["x"])
            yc = float(wg["y_gap"])
            gw = float(wg["gap_width"])
            half = gw * 0.5
            y_lo = max(0.0, yc - half)
            y_hi = min(rh, yc + half)
            if y_lo > 0.0:
                p0 = self._world_to_screen(xw, 0.0)
                p1 = self._world_to_screen(xw, y_lo)
                pygame.draw.line(self._canvas, self._wall, p0, p1, self._wall_line_px)
            if y_hi < rh:
                p0 = self._world_to_screen(xw, y_hi)
                p1 = self._world_to_screen(xw, rh)
                pygame.draw.line(self._canvas, self._wall, p0, p1, self._wall_line_px)

        if len(corners) >= 3:
            poly = [self._world_to_screen(float(px), float(py)) for px, py in corners]
            pygame.draw.polygon(self._canvas, self._figure_fill, poly)
            pygame.draw.polygon(self._canvas, self._figure_edge, poly, 2)

        self._draw_rays(self._canvas, env_state)

    def present(self) -> None:
        """Blit canvas to the window and flip buffers (human mode only)."""
        if self._mode != "human":
            return
        if self._screen is None:
            self._screen = pygame.display.set_mode((self._surf_w, self._surf_h))
        self._screen.blit(self._canvas, (0, 0))
        pygame.display.flip()
        pygame.event.pump()

    def render(self, env_state: dict) -> np.ndarray | None:
        """
        Draw current state. Returns RGB array or None.

        For ``human`` mode, calls :meth:`draw_world` then :meth:`present`.
        """
        self.draw_world(env_state)
        if self._mode == "human":
            self.present()
            return None

        if self._mode == "rgb_array":
            arr = pygame.surfarray.array3d(self._canvas)
            return np.transpose(arr, (1, 0, 2)).copy()

        return None

    def close(self) -> None:
        """Destroy pygame window and free resources."""
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None
