import numpy as np
import gymnasium
from gymnasium import spaces
import pymunk

from src.utils.config import Config
from src.envs.figure import TFigure
from src.envs.maze import Maze
from src.envs.renderer import Renderer
from src.utils.geometry import cast_rays_detailed


class MazeEnv(gymnasium.Env):
    """
    Gymnasium environment for the figure-navigation task.
    Strictly follows the 2D action space: (delta_theta, f).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Config, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # Init physics space
        self.space = pymunk.Space()
        self.maze = Maze(config, self.space)
        self.figure = TFigure(self.space, config)

        # Action space: [delta_theta (deg), f (thrust)]
        self.action_space = spaces.Box(
            low=np.array([-config.max_delta_theta, -config.max_thrust], dtype=np.float32),
            high=np.array([config.max_delta_theta, config.max_thrust], dtype=np.float32),
            dtype=np.float32
        )

        # Calculate number of rays: 12 corners * n_ray_directions
        self.n_corners = len(self.figure._outline_local)
        self.k_rays = self.n_corners * config.n_ray_directions
        
        # State: x, y, theta + K ray distances
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(3 + self.k_rays,), 
            dtype=np.float32
        )

        self.renderer = Renderer(config, render_mode=render_mode) if render_mode else None

        # Internal state tracking
        self.step_count = 0
        self._rho_1 = 0.0
        self._rho_2 = 0.0
        self._last_rays = {}  # for rendering

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.step_count = 0
        self.maze.randomise_gaps(self.np_random)

        # Spawn figure on the left side of the room
        start_x = self.config.room_width * 0.1
        start_y = self.config.room_height * 0.5
        self.figure.set_state(start_x, start_y, 0.0)

        # Initial progress
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

        # Clip action to valid bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Save current state for potential rollback
        old_x, old_y = self.figure.body.position
        old_theta = self.figure.body.angle

        # Exact formulas from Section 4 of the specification
        delta_theta_deg, f = action[0], action[1]
        delta_theta_rad = np.radians(delta_theta_deg)
        
        new_theta = old_theta + delta_theta_rad
        new_x = old_x + f * np.cos(new_theta)
        new_y = old_y + f * np.sin(new_theta)

        # Apply candidate state
        self.figure.set_state(new_x, new_y, new_theta)
        self.space.reindex_shapes_for_body(self.figure.body)

        # Check Predicates C(s_tilde) and B(s_tilde)
        collision = False
        for shape in self.figure._shapes:
            query = self.space.shape_query(shape)
            for hit in query:
                # Ignore self-collisions
                if hit.shape.body != self.figure.body:
                    collision = True
                    break
            if collision: break

        corners = self.figure.get_corners()
        out_of_bounds = self.maze.is_out_of_bounds(corners)

        # Rollback if necessary
        if collision or out_of_bounds:
            self.figure.set_state(old_x, old_y, old_theta)
            self.space.reindex_shapes_for_body(self.figure.body)
            corners = self.figure.get_corners() # Recompute corners for rays after rollback

        # Compute new state and reward
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
        theta = self.figure.body.angle
        corners = self.figure.get_corners()

        # Fixed set of local directions for rays, relative to figure's angle
        base_dirs = np.linspace(0, 2 * np.pi, self.config.n_ray_directions, endpoint=False)
        directions = (base_dirs + theta).tolist()

        distances, endpoints, hits = cast_rays_detailed(
            origins=corners,
            directions=directions,
            space=self.space,
            r_max=self.config.r_max,
            ignore_bodies=[self.figure.body]
        )

        # Cache for rendering
        self._last_rays = {
            "origins": np.repeat(corners, len(directions), axis=0).tolist(),
            "endpoints": endpoints,
            "hits": hits
        }

        state = np.zeros(3 + len(distances), dtype=np.float32)
        state[0], state[1], state[2] = x, y, theta
        state[3:] = distances
        return state

    def _compute_reward(self, s, s_next, collision: bool, out_of_bounds: bool) -> float:
        walls = self.maze.get_wall_geometries()
        new_rho_1 = self.figure.compute_progress(walls[0])
        new_rho_2 = self.figure.compute_progress(walls[1])

        delta_rho_1 = new_rho_1 - self._rho_1
        delta_rho_2 = new_rho_2 - self._rho_2

        # Update stored progress
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
            "ray_origins": self._last_rays.get("origins", []),
            "ray_endpoints": self._last_rays.get("endpoints", []),
            "ray_hits": self._last_rays.get("hits", [])
        }
        return self.renderer.render(env_state)

    def seed(self, seed: int) -> None:
        self.np_random = np.random.default_rng(seed)

    def close(self) -> None:
        if self.renderer:
            self.renderer.close()