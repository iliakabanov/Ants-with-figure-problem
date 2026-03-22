import numpy as np
import pygame

from src.utils.config import Config
from src.envs.maze_env import MazeEnv


def keys_to_action(keys, config: Config) -> np.ndarray:
    """
    Convert pressed keys to a continuous action vector.
    Strictly follows 2D action space: (delta_theta, thrust_f)
    """
    delta_theta = 0.0
    f = 0.0

    # Вращение (Left/Right)
    if keys[pygame.K_LEFT]:
        delta_theta = config.max_delta_theta
    if keys[pygame.K_RIGHT]:
        delta_theta = -config.max_delta_theta

    # Вперед/Назад по текущему курсу (Up/Down)
    if keys[pygame.K_UP]:
        f = config.max_thrust
    if keys[pygame.K_DOWN]:
        f = -config.max_thrust

    return np.array([delta_theta, f], dtype=np.float32)


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
    print("Controls (Strictly by spec):")
    print("  UP/DOWN    - Thrust (push along heading)")
    print("  LEFT/RIGHT - Rotate")
    print("  R: Reset, ESC: Quit")

    while running:
        # 1. Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
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
        draw_state = {
            "room_width": config.room_width,
            "room_height": config.room_height,
            "wall_geometries": env.unwrapped.maze.get_wall_geometries(),
            "figure_corners": env.unwrapped.figure.get_corners(),
            "ray_origins": env.unwrapped._last_rays.get("origins", []),
            "ray_endpoints": env.unwrapped._last_rays.get("endpoints", []),
            "ray_hits": env.unwrapped._last_rays.get("hits", []),
        }
        if config.viz_corner_labels:
            draw_state["figure_corner_labels"] = [str(lb) for lb in env.unwrapped.figure.corner_labels]
        env.unwrapped.renderer.draw_world(draw_state)
        
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
    cfg.viz_corner_labels = True
    run_interactive(cfg)