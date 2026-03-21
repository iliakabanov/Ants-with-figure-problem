from src.utils.config import Config


def evaluate(config: Config, agent,
             n_episodes: int = 100) -> tuple[float, float, float, float]:
    """
    Run deterministic evaluation episodes and report metrics.

    Args:
        config:     Config with all hyperparameters.
        agent:      trained ActorCriticAgent or ReinforceAgent instance.
        n_episodes: number of evaluation episodes to run.

    Returns:
        mean_return:  mean cumulative reward across episodes.
        success_rate: fraction of episodes where the figure reached the goal.
        mean_steps:   mean number of steps per episode.
        std_return:   standard deviation of cumulative reward across episodes.
    """
    pass


if __name__ == "__main__":
    """
    Evaluate a trained model.
    Usage: python run/eval.py --agent ac
                              --checkpoint artifacts/checkpoints/ac_ep5000.pt
                              --n_episodes 100
    Args:
        --agent:      'ac' or 'reinforce' (required).
        --checkpoint: path to saved checkpoint (required).
        --n_episodes: number of evaluation episodes (default: 100).
    Stdout: prints mean return ± std, success rate, mean episode length.
    Saves:  artifacts/logs/eval_{agent}.csv
            columns: episode, return, steps, success.
    """
    pass
