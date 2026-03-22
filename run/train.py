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


def _grad_norm(agent) -> float:
    """Compute L2 norm of actor gradients after the last update."""
    total = 0.0
    for p in agent.actor.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total ** 0.5


def build_agent(config: Config,
                agent_name: str):
    """
    Instantiate the requested agent.

    Args:
        config:        Config with all hyperparameters.
        agent_name:    'ac' or 'reinforce'.
        baseline_name: 'zero' or 'mean_return'.
                       Ignored when agent_name is 'ac'.

    Returns:
        agent: fully initialised agent.
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _env = MazeEnv(config)
    state_dim = _env.observation_space.shape[0]
    action_dim = _env.action_space.shape[0]
    _env.close()

    if agent_name == 'ac':
        return ActorCriticAgent(state_dim, action_dim, config, device)

    elif agent_name == 'reinforce':
        baseline = ZeroBaseline()
        return ReinforceAgent(state_dim, action_dim, config, device, baseline)

    else:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose 'ac' or 'reinforce'.")


def _run_eval(config: Config, agent, n_episodes: int = 20):
    """Run n_episodes deterministic episodes, return (mean_return, success_rate)."""
    env = MazeEnv(config)
    returns, successes = [], []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        ep_return, done, info = 0.0, False, {}
        while not done:
            result = agent.select_action(state, deterministic=True)
            action = result[0] if isinstance(result, tuple) else result
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
        successes.append(float(info.get('rho_1', 0.0) >= 1.0 and
                               info.get('rho_2', 0.0) >= 1.0))
    env.close()
    return float(np.mean(returns)), float(np.mean(successes))


def train(config: Config, agent) -> None:
    """
    Shared training loop for both agents.
    Actor-Critic:  calls agent.update() after every env step.
    REINFORCE:     calls agent.update() once per completed episode.

    Logging: appends one row per episode to artifacts/logs/{agent}.csv
             with columns: episode, return, steps, success.
    Eval:    every config.eval_every episodes, runs 20 deterministic
             episodes and prints mean return and success rate to stdout.
    Checkpoints: saved every config.checkpoint_every episodes to
                 artifacts/checkpoints/{agent}_ep{N}.pt.

    Args:
        config: Config with all hyperparameters.
        agent:  ActorCriticAgent or ReinforceAgent instance.
    """
    is_ac = isinstance(agent, ActorCriticAgent)
    agent_tag = 'ac' if is_ac else 'reinforce'

    os.makedirs('artifacts/checkpoints', exist_ok=True)
    os.makedirs('artifacts/logs', exist_ok=True)

    log_path = f'artifacts/logs/{agent_tag}.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['episode', 'return', 'steps', 'success'])

    env = MazeEnv(config)

    grad_norms: list[float] = []

    for episode in range(1, config.n_episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        ep_return, ep_steps, done, info = 0.0, 0, False, {}
        ep_grad_norms: list[float] = []

        while not done:
            if is_ac:
                action = agent.select_action(state)
                log_prob = None
            else:
                action, log_prob = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_steps += 1

            if is_ac:
                agent.replay_buffer.push(state, action, reward, next_state, done)
                if len(agent.replay_buffer) > 0:
                    agent.update()
                    ep_grad_norms.append(_grad_norm(agent))
            else:
                agent.store_transition(state, action, reward, log_prob)

            state = next_state

        if not is_ac:
            agent.update()
            ep_grad_norms.append(_grad_norm(agent))

        if ep_grad_norms:
            grad_norms.append(float(np.mean(ep_grad_norms)))

        success = int(info.get('rho_1', 0.0) >= 1.0 and
                      info.get('rho_2', 0.0) >= 1.0)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, ep_return, ep_steps, success])

        if episode % config.eval_every == 0:
            mean_ret, success_rate = _run_eval(config, agent)
            mean_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
            grad_norms.clear()
            print(f"ep {episode:5d} | eval return: {mean_ret:8.2f} | success: {success_rate:.2f} | grad_norm: {mean_grad:.4f}")

        if episode % config.checkpoint_every == 0:
            path = f'artifacts/checkpoints/{agent_tag}_ep{episode}.pt'
            agent.save(path)
            print(f"checkpoint -> {path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Actor-Critic or REINFORCE agent.")
    parser.add_argument('--agent',    required=True, choices=['ac', 'reinforce'])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    config = Config()
    if args.seed is not None:
        config.seed = args.seed

    agent = build_agent(config, args.agent)
    train(config, agent)
