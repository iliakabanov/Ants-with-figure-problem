# Ants-with-figure-problem

## Problem Definition

A rigid T-shaped figure must be navigated from the left side of a rectangular room to the right side, passing through two pairs of narrow vertical wall gaps. The figure cannot be deformed; it must be physically manoeuvred through each gap by rotating and translating it. An agent controls two continuous actions at each timestep: a rotation increment and a thrust force along the figure's axis. The episode ends when the figure fully clears both walls (success) or when a collision, out-of-bounds event, or step limit is reached (failure).

## Environment Specification

- **Room**: `room_width × room_height` (default 26 × 17 units).
- **Figure**: T-shape with top bar (5×1), leg (4×1), and bottom bar (2×1). Total area is fixed.
- **Walls**: Two pairs of vertical walls at fixed x-coordinates, each with a horizontal gap of width `gap_width` (default 3.0 units).
- **State**: `(x, y, theta, d_1, ..., d_K)` — centre-of-mass position, orientation angle, and K ray distances cast from figure corners in `n_ray_directions` directions, clipped to `r_max`.
- **Action**: `(delta_theta, f)` — rotation increment in `[-max_delta_theta, max_delta_theta]` degrees and thrust force in `[-max_thrust, max_thrust]`.
- **Reward**: `r_fin·1_fin − r_col·1_col − r_oob·1_oob + r_wall·(Δρ_1 + Δρ_2) − r_step`, where ρ_i(s) = S_i_wall(s) / S_total is the fraction of the figure's area that has passed wall i.
- **Termination**: success (figure past both walls), collision, out-of-bounds, or `max_steps` exceeded.
- **Randomisation**: gap centre positions are sampled uniformly each episode within `[gap_margin, room_height − gap_margin]` when `randomise_gaps=True`.

## Reproducibility

All stochastic elements are seeded via `Config.seed` (default 42). Pass `--seed` to `run/train.py` or `run/eval.py` to override. The environment exposes `MazeEnv.seed(seed)` and `reset(seed=...)` for explicit seeding. Gap randomisation uses a numpy `Generator` passed to `Maze.randomise_gaps(rng)`.

To reproduce a training run:

```bash
python run/train.py --agent ac --seed 42
python run/train.py --agent reinforce --baseline mean_return --seed 42
```

To reproduce evaluation:

```bash
python run/eval.py --agent ac --checkpoint artifacts/checkpoints/ac_ep5000.pt --seed 42
```

Checkpoints are saved to `artifacts/checkpoints/` and logs to `artifacts/logs/`.
