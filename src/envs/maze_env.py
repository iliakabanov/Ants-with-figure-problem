import numpy as np
import gymnasium
from gymnasium import spaces
import pymunk

from src.utils.config import Config, MAZE_ACTION_DIM
from src.envs.figure import TFigure
from src.envs.maze import Maze
from src.envs.renderer import Renderer


# Два вертикальных ряда стен с проёмами; наблюдение добавляет 2*(Δx, Δy к центру щели) + ширина щели.
N_WALL_PAIRS = 2
# pose (3) + относительные (x_wall - px, y_gap - py) на каждую пару + gap_width
OBSERVATION_DIM = 3 + N_WALL_PAIRS * 2 + 1


class MazeEnv(gymnasium.Env):
    """
    Gymnasium environment for the figure-navigation task.
    Action: (fx_body, fy_body, delta_theta_deg) — сдвиг ЦМ в системе тела (+x вдоль верхней перекладины Т,
    +y вдоль ножки), затем в мир через ``Vec2d(fx,fy).rotated(theta)``; третий компонент — поворот в градусах.

    Observation: ``(x, y, θ)`` фигуры в мире; для каждой пары стен — смещение от ЦМ до центра проёма
    ``(x_wall - x, y_gap - y)``; последний скаляр — ``gap_width`` (общий для всех проёмов).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Config, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        self.space = pymunk.Space()
        self.maze = Maze(config, self.space)
        self.figure = TFigure(self.space, config)

        mt = float(config.max_thrust)
        self.action_space = spaces.Box(
            low=np.array([-mt, -mt, -config.max_delta_theta], dtype=np.float32),
            high=np.array([mt, mt, config.max_delta_theta], dtype=np.float32),
            dtype=np.float32,
        )
        if self.action_space.shape[0] != MAZE_ACTION_DIM:
            raise ValueError(
                f"MazeEnv action_space must have shape ({MAZE_ACTION_DIM},); got {self.action_space.shape}."
            )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        self.renderer = Renderer(config, render_mode=render_mode) if render_mode else None

        self.step_count = 0
        self._rho_1 = 0.0
        self._rho_2 = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.step_count = 0
        self.maze.randomise_gaps(self.np_random)

        start_x = self.config.room_width * 0.2
        start_y = self.config.room_height * 0.5
        start_theta = 0.0
        self.figure.set_state(start_x, start_y, start_theta)
        self.space.reindex_shapes_for_body(self.figure.body)
        self.space.reindex_static()

        walls = self.maze.get_wall_geometries()
        self._rho_1 = self.figure.compute_progress(walls[0])
        self._rho_2 = self.figure.compute_progress(walls[1])

        state = self._compute_state()
        info = {
            "rho_1": self._rho_1,
            "rho_2": self._rho_2,
            "collision": False,
            "out_of_bounds": False,
        }
        return state, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)

        old_x, old_y = self.figure.body.position
        old_theta = self.figure.body.angle

        fx_b, fy_b, delta_theta_deg = float(action[0]), float(action[1]), float(action[2])
        delta_theta_rad = np.radians(delta_theta_deg)

        delta_world = pymunk.Vec2d(fx_b, fy_b).rotated(old_theta)
        new_x = old_x + delta_world.x
        new_y = old_y + delta_world.y
        new_theta = old_theta + delta_theta_rad

        self.figure.set_state(new_x, new_y, new_theta)
        self.space.reindex_shapes_for_body(self.figure.body)
        self.space.reindex_static()

        collision = False
        for shape in self.figure._shapes:
            query = self.space.shape_query(shape)
            for hit in query:
                if hit.shape.body != self.figure.body:
                    collision = True
                    break
            if collision:
                break

        corners = self.figure.get_corners()
        out_of_bounds = self.maze.is_out_of_bounds(corners)

        if collision or out_of_bounds:
            self.figure.set_state(old_x, old_y, old_theta)
            self.space.reindex_shapes_for_body(self.figure.body)
            self.space.reindex_static()
            corners = self.figure.get_corners()

        delta_x_world = float(self.figure.body.position.x - old_x)
        state = self._compute_state()
        reward = self._compute_reward(None, state, collision, out_of_bounds, delta_x_world)

        terminated = bool(self._rho_1 >= 1.0 and self._rho_2 >= 1.0)
        truncated = bool(self.step_count >= self.config.max_steps)

        info = {
            "rho_1": self._rho_1,
            "rho_2": self._rho_2,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
        }

        return state, reward, terminated, truncated, info

    def _compute_state(self) -> np.ndarray:
        x = float(self.figure.body.position.x)
        y = float(self.figure.body.position.y)
        theta = float(self.figure.body.angle)
        walls = self.maze.get_wall_geometries()
        gw = float(self.config.gap_width)
        parts: list[float] = [x, y, theta]
        for w in walls:
            parts.append(float(w["x"]) - x)
            parts.append(float(w["y_gap"]) - y)
        parts.append(gw)
        out = np.asarray(parts, dtype=np.float32)
        if out.shape[0] != OBSERVATION_DIM:
            raise RuntimeError(
                f"Observation dim {out.shape[0]} != OBSERVATION_DIM={OBSERVATION_DIM}."
            )
        return out

    def _compute_reward(
        self,
        s,
        s_next,
        collision: bool,
        out_of_bounds: bool,
        delta_x_world: float,
    ) -> float:
        walls = self.maze.get_wall_geometries()
        new_rho_1 = self.figure.compute_progress(walls[0])
        new_rho_2 = self.figure.compute_progress(walls[1])

        delta_rho_1 = new_rho_1 - self._rho_1
        delta_rho_2 = new_rho_2 - self._rho_2

        self._rho_1 = new_rho_1
        self._rho_2 = new_rho_2

        reward = 0.0
        if self._rho_1 >= 1.0 and self._rho_2 >= 1.0:
            reward += self.config.r_fin

        if collision:
            reward -= self.config.r_col
        if out_of_bounds:
            reward -= self.config.r_oob

        reward += self.config.r_wall * (delta_rho_1 + delta_rho_2)
        rr = float(self.config.r_right)
        if rr != 0.0:
            reward += rr * max(0.0, delta_x_world)
        reward -= self.config.r_step

        return reward

    def render(self) -> np.ndarray | None:
        if self.renderer is None:
            return None

        env_state = {
            "room_width": self.config.room_width,
            "room_height": self.config.room_height,
            "wall_geometries": self.maze.get_wall_geometries(),
            "figure_corners": self.figure.get_corners(),
            "ray_origins": [],
            "ray_endpoints": [],
            "ray_hits": [],
        }
        return self.renderer.render(env_state)

    def seed(self, seed: int) -> None:
        self.np_random = np.random.default_rng(seed)

    def close(self) -> None:
        if self.renderer:
            self.renderer.close()
