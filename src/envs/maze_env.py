import numpy as np
import gymnasium
from gymnasium import spaces
import pymunk

from src.utils.config import Config
from src.envs.figure import TFigure
from src.envs.maze import Maze
from src.envs.renderer import Renderer


class MazeEnv(gymnasium.Env):
    """
    Gymnasium environment for the figure-navigation task.
    Action: (fx_body, fy_body, delta_theta_deg) — сдвиг ЦМ в системе тела (+x вдоль верхней перекладины Т,
    +y вдоль ножки), затем в мир через ``Vec2d(fx,fy).rotated(theta)``; третий компонент — поворот в градусах.

    State: (x, y, theta, wall1_x, gap1_lo, gap1_hi, wall2_x, gap2_lo, gap2_hi) — 9 чисел.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # State dim: x, y, theta + (wall_x, gap_lo, gap_hi) × 2 walls = 9
    STATE_DIM = 9

    def __init__(self, config: Config, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # Init physics space
        self.space = pymunk.Space()
        self.maze = Maze(config, self.space)
        self.figure = TFigure(self.space, config)

        mt = float(config.max_thrust)
        self.action_space = spaces.Box(
            low=np.array([-mt, -mt, -config.max_delta_theta], dtype=np.float32),
            high=np.array([mt, mt, config.max_delta_theta], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.STATE_DIM,),
            dtype=np.float32,
        )

        self.renderer = Renderer(config, render_mode=render_mode) if render_mode else None

        # Bounding radius: max distance from COM to any outline vertex.
        # Used as spawn margin so the figure never intersects room walls at reset.
        pts = np.array(self.figure._outline_local)
        self._spawn_margin = float(np.max(np.linalg.norm(pts, axis=1)))

        self.step_count = 0
        self._rho_1 = 0.0
        self._rho_2 = 0.0
        self._prev_x = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.step_count = 0
        self.maze.randomise_gaps(self.np_random)

        m = self._spawn_margin
        start_x = self.config.room_width * 0.2
        if self.config.randomise_y:
            start_y = float(self.np_random.uniform(m, self.config.room_height - m))
        else:
            start_y = self.config.room_height * 0.5
        
        if self.config.randomise_theta:
            start_theta = float(self.np_random.uniform(-np.pi, np.pi))
        else:
            start_theta = 0.0
        self.figure.set_state(start_x, start_y, start_theta)
        self.space.reindex_shapes_for_body(self.figure.body)
        self.space.reindex_static()
        self._prev_x = start_x

        walls = self.maze.get_wall_geometries()
        self._rho_1 = self.figure.compute_progress(walls[0])
        self._rho_2 = self.figure.compute_progress(walls[1])

        state = self._compute_state()
        info = {
            "rho_1": self._rho_1,
            "rho_2": self._rho_2,
            "collision": False,
            "out_of_bounds": False
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
            if collision: break

        corners = self.figure.get_corners()
        out_of_bounds = self.maze.is_out_of_bounds(corners)

        if collision or out_of_bounds:
            self.figure.set_state(old_x, old_y, old_theta)
            self.space.reindex_shapes_for_body(self.figure.body)
            self.space.reindex_static()
            corners = self.figure.get_corners()

        state = self._compute_state()
        reward = self._compute_reward(None, state, collision, out_of_bounds)

        terminated = bool(self._rho_1 >= 1.0 and self._rho_2 >= 1.0)
        truncated = bool(self.step_count >= self.config.max_steps)

        info = {
            "rho_1": self._rho_1,
            "rho_2": self._rho_2,
            "collision": collision,
            "out_of_bounds": out_of_bounds
        }

        return state, reward, terminated, truncated, info

    def _compute_state(self) -> np.ndarray:
        x, y = self.figure.body.position
        theta = float(self.figure.body.angle)

        walls = self.maze.get_wall_geometries()
        half_gap = self.config.gap_width * 0.5

        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        state[0], state[1], state[2] = x, y, theta
        for i, wg in enumerate(walls):
            base = 3 + i * 3
            state[base]     = wg['x']
            state[base + 1] = wg['y_gap'] - half_gap  # gap lower bound
            state[base + 2] = wg['y_gap'] + half_gap  # gap upper bound
        return state

    def _compute_reward(self, s, s_next, collision: bool, out_of_bounds: bool) -> float:
        walls = self.maze.get_wall_geometries()
        new_rho_1 = self.figure.compute_progress(walls[0])
        new_rho_2 = self.figure.compute_progress(walls[1])

        delta_rho_1 = new_rho_1 - self._rho_1
        delta_rho_2 = new_rho_2 - self._rho_2

        self._rho_1 = new_rho_1
        self._rho_2 = new_rho_2

        cur_x = float(self.figure.body.position.x)
        delta_x = cur_x - self._prev_x
        self._prev_x = cur_x

        reward = 0.0
        if self._rho_1 >= 1.0 and self._rho_2 >= 1.0:
            reward += self.config.r_fin

        if collision:
            reward += self.config.r_col
        if out_of_bounds:
            reward += self.config.r_oob

        reward += self.config.r_wall * (delta_rho_1 + delta_rho_2)
        reward += self.config.r_progress * delta_x
        reward += self.config.r_step

        return reward

    def render(self) -> np.ndarray | None:
        if self.renderer is None:
            return None
        
        env_state = {
            "room_width": self.config.room_width,
            "room_height": self.config.room_height,
            "wall_geometries": self.maze.get_wall_geometries(),
            "figure_corners": self.figure.get_corners(),
        }
        return self.renderer.render(env_state)

    def seed(self, seed: int) -> None:
        self.np_random = np.random.default_rng(seed)

    def close(self) -> None:
        if self.renderer:
            self.renderer.close()