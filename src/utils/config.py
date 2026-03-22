from dataclasses import dataclass


@dataclass
class Config:
    """All hyperparameters and environment settings."""

    # Environment
    room_width: float = 30.0
    room_height: float = 20.0
    gap_width: float = 4.0          # fixed gap width, same for all wall pairs
    gap_margin: float = 6.0
    wall_radius: float = 0.0        # pymunk segment radius for interior walls
    # Minimum distance from gap centre to floor/ceiling.
    # When gaps are randomised, y_gap is sampled uniformly from
    # [gap_margin, room_height - gap_margin].
    # This prevents degenerate layouts where the gap is flush against
    # a room wall, leaving no space for the figure to manoeuvre.
    # Value 4.0 > half the figure height (6/2 = 3) so the figure
    # can always approach the gap from either side.
    corridor_length: float = 7.0
    randomise_gaps: bool = False    # randomise y_gap each episode; False = gaps fixed at room centre
    randomise_theta: bool = False   # randomise initial figure angle each episode
    randomise_y: bool = False       # randomise initial y position each episode
    max_steps: int = 100
    r_max: float = 10.0
    # MazeEnv не использует: с каждого угла 5 лучей по розе вокруг ``FigureCornerLabel.wind8``.
    n_ray_directions: int = 8

    # Figure
    top_bar_length: float = 1.0 #5.0
    leg_length: float = 1.0 #5.0
    bottom_bar_length: float = 1 #1.5
    thickness: float = 1.0 #1.0

    # Actions: (fx_body, fy_body, delta_theta_deg); +x тела — вдоль верхней перекладины Т, +y — ортогонально
    max_delta_theta: float = 45.0
    max_thrust: float = 1

    # Reward
    r_fin: float = 100.0
    r_col: float = -0.0
    r_oob: float = -10.0
    r_wall: float = 1000.0
    r_step: float = -1.0
    r_progress: float = 0.1    # scale for delta_x reward (positive = right, negative = left)

    # Actor-Critic
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    hidden_dim: int = 256
    warmup_steps: int = 1000

    # REINFORCE
    lr_reinforce: float = 10**(-3)
    grad_clip: float = 100.0         # max gradient norm; 0.0 = no clipping
    baseline: str = 'zero'          # 'zero' | 'mean_return'
    baseline_momentum: float = 0.99

    # Training
    n_episodes: int = 100
    eval_every: int = 5
    checkpoint_every: int = 5
    seed: int = 42
