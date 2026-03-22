import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv

import numpy as np

from src.utils.config import Config
from src.envs.maze_env import MazeEnv
from src.agents.actor_critic_agent import ActorCriticAgent
from src.agents.reinforce_agent import ReinforceAgent
from src.agents.baselines import ZeroBaseline


def evaluate(config: Config, agent,
             n_episodes: int = 20,
             render_mode: str | None = None) -> tuple[float, float, float, float]:
    """
    Run deterministic evaluation episodes and report metrics.

    Args:
        config:      Config with all hyperparameters.
        agent:       trained ActorCriticAgent or ReinforceAgent instance.
        n_episodes:  number of evaluation episodes to run.
        render_mode: 'human' to open a render window, None to run headless.

    Returns:
        mean_return:  mean cumulative reward across episodes.
        success_rate: fraction of episodes where the figure reached the goal.
        mean_steps:   mean number of steps per episode.
        std_return:   standard deviation of cumulative reward across episodes.
    """
    env = MazeEnv(config, render_mode=render_mode)
    returns, steps_list, successes = [], [], []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        ep_return, ep_steps, done, info = 0.0, 0, False, {}

        while not done:
            result = agent.select_action(state, deterministic=True)
            action = result[0] if isinstance(result, tuple) else result
            state, reward, terminated, truncated, info = env.step(action)
            if render_mode == 'human':
                env.render()
            done = terminated or truncated
            ep_return += reward
            ep_steps += 1

        returns.append(ep_return)
        steps_list.append(ep_steps)
        successes.append(float(
            info.get('rho_1', 0.0) >= 1.0 and info.get('rho_2', 0.0) >= 1.0
        ))

    env.close()

    return (
        float(np.mean(returns)),
        float(np.mean(successes)),
        float(np.mean(steps_list)),
        float(np.std(returns)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained agent.")
    parser.add_argument('--agent',      required=True, choices=['ac', 'reinforce'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seed',       type=int, default=None)
    parser.add_argument('--render',     action='store_true',
                        help='Render episodes in a window during evaluation.')
    args = parser.parse_args()

    import torch
    config = Config()
    if args.seed is not None:
        config.seed = args.seed

    render_mode = 'human' if args.render else None

    env_tmp = MazeEnv(config)
    state_dim  = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    env_tmp.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.agent == 'ac':
        agent = ActorCriticAgent(state_dim, action_dim, config, device)
    else:
        agent = ReinforceAgent(state_dim, action_dim, config, device, ZeroBaseline())

    agent.load(args.checkpoint)

    mean_ret, success_rate, mean_steps, std_ret = evaluate(
        config, agent, n_episodes=args.n_episodes, render_mode=render_mode
    )

    print(f"Mean return:   {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"Success rate:  {success_rate:.2%}")
    print(f"Mean steps:    {mean_steps:.1f}")

    os.makedirs('artifacts/logs', exist_ok=True)
    log_path = f'artifacts/logs/eval_{args.agent}.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'return', 'steps', 'success'])

    env = MazeEnv(config)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for ep in range(args.n_episodes):
            state, _ = env.reset(seed=ep)
            ep_return, ep_steps, done, info = 0.0, 0, False, {}
            while not done:
                result = agent.select_action(state, deterministic=True)
                action = result[0] if isinstance(result, tuple) else result
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_return += reward
                ep_steps += 1
            success = int(info.get('rho_1', 0.0) >= 1.0 and info.get('rho_2', 0.0) >= 1.0)
            writer.writerow([ep + 1, ep_return, ep_steps, success])
    env.close()

    print(f"Saved -> {log_path}")
