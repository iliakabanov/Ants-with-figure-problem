from dataclasses import dataclass

# Continuous action size for MazeEnv / agents (fx_body, fy_body, delta_theta_deg).
MAZE_ACTION_DIM = 3


@dataclass
class Config:
    """All hyperparameters and environment settings."""

    # Environment
    room_width: float = 30.0
    room_height: float = 20.0
    gap_width: float = 4.0          # fixed gap width, same for all wall pairs
    gap_margin: float = 4.0
    wall_radius: float = 0.0        # pymunk segment radius for interior walls
    # Minimum distance from gap centre to floor/ceiling.
    # When gaps are randomised, y_gap is sampled uniformly from
    # [gap_margin, room_height - gap_margin].
    # This prevents degenerate layouts where the gap is flush against
    # a room wall, leaving no space for the figure to manoeuvre.
    # Value 4.0 > half the figure height (6/2 = 3) so the figure
    # can always approach the gap from either side.
    corridor_length: float = 7.0
    randomise_gaps: bool = True     # randomise y_gap each episode
    max_steps: int = 2000

    # Figure
    top_bar_length: float = 5.0
    leg_length: float = 5.0
    bottom_bar_length: float = 1.5
    thickness: float = 1.0

    # Actions: (fx_body, fy_body, delta_theta_deg); +x тела — вдоль верхней перекладины Т, +y — ортогонально
    max_delta_theta: float = 5.0
    max_thrust: float = 0.1

    # Reward
    r_fin: float = 100.0
    r_col: float = 10.0
    r_oob: float = 10.0
    r_wall: float = 10.0
    r_step: float = 1.0
    # Бонус за сдвиг в мировую сторону +x (вправо) за успешный шаг; 0 = выключено.
    r_right: float = 1.0

    # Actor-Critic (replay + target Q + SAC-style entropy regularisation)
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    # Weight on log pi in actor loss and soft Bellman target; 0 disables entropy bonus.
    entropy_coef: float = 0.01
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_capacity: int = 200_000
    hidden_dim: int = 128
    warmup_steps: int = 0

    # REINFORCE
    lr_reinforce: float = 1e-3
    baseline: str = 'zero'          # 'zero' | 'mean_return'
    baseline_momentum: float = 0.99

    # Training
    n_episodes: int = 5000
    eval_every: int = 50
    checkpoint_every: int = 500
    seed: int = 42
