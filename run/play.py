import numpy as np
import pygame

from src.utils.config import Config
from src.envs.maze_env import MazeEnv


def keys_to_action(keys, config: Config) -> np.ndarray:
    """(fx_body, fy_body, delta_theta). W/S — поперечина; E/Q — ножка (±fy); A/D — поворот."""
    fx = 0.0
    fy = 0.0
    delta_theta = 0.0
    mt = config.max_thrust

    if keys[pygame.K_w]:
        fx += mt
    if keys[pygame.K_s]:
        fx -= mt
    if keys[pygame.K_e]:
        fy += mt
    if keys[pygame.K_q]:
        fy -= mt
    if keys[pygame.K_a]:
        delta_theta = config.max_delta_theta
    if keys[pygame.K_d]:
        delta_theta = -config.max_delta_theta

    return np.array([fx, fy, delta_theta], dtype=np.float32)


def render_hud(surface: pygame.Surface, step: int, total_reward: float, rho_1: float, rho_2: float) -> None:
    """
    Draw heads-up display overlay on the pygame surface.
    """
    font = pygame.font.SysFont("Courier", 18, bold=True)
    text_lines = [
        f"Step:   {step}",
        f"Reward: {total_reward:.2f}",
        f"Rho 1:  {rho_1 * 100:.1f}%",
        f"Rho 2:  {rho_2 * 100:.1f}%"
    ]
    
    y_offset = 10
    for line in text_lines:
        text_surf = font.render(line, True, (20, 20, 20))
        surface.blit(text_surf, (10, y_offset))
        y_offset += 25


def run_interactive(config: Config) -> None:
    # Initialize environment with human rendering mode
    env = MazeEnv(config, render_mode="human")
    obs, info = env.reset()

    clock = pygame.time.Clock()
    running = True

    step = 0
    total_reward = 0.0
    rho_1, rho_2 = info["rho_1"], info["rho_2"]

    print("=== Figure Maze Manual Play ===")
    print("Controls:")
    print("  W S — поперечина (+x тела), E / Q — ножка (+y / −y)")
    print("  A / D — поворот (+ / − max_delta_theta)")
    print("  R: Reset, ESC: Quit")

    while running:
        # 1. Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    step = 0
                    total_reward = 0.0
                    rho_1, rho_2 = info["rho_1"], info["rho_2"]
                    print("Environment reset.")

        # 2. Read keyboard and map to action
        keys = pygame.key.get_pressed()
        action = keys_to_action(keys, config)

        # 3. Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        total_reward += reward
        rho_1, rho_2 = info["rho_1"], info["rho_2"]

        # 4 & 5. Render to back buffer, draw HUD, then present
        env.unwrapped.renderer.draw_world({
            "room_width": config.room_width,
            "room_height": config.room_height,
            "wall_geometries": env.unwrapped.maze.get_wall_geometries(),
            "figure_corners": env.unwrapped.figure.get_corners(),
            "ray_origins": env.unwrapped._last_rays.get("origins", []),
            "ray_endpoints": env.unwrapped._last_rays.get("endpoints", []),
            "ray_hits": env.unwrapped._last_rays.get("hits", []),
        })
        
        render_hud(env.unwrapped.renderer.canvas, step, total_reward, rho_1, rho_2)
        env.unwrapped.renderer.present()

        if terminated or truncated:
            reason = "Finished!" if terminated else "Time out!"
            print(f"Episode ended ({reason}). Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            step = 0
            total_reward = 0.0

        # Cap framerate to real-time 30 FPS
        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    cfg = Config()
    # For manual play we can optionally increase time limit to allow slow exploration
    cfg.max_steps = 5000
    run_interactive(cfg)