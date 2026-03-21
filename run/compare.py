from src.utils.config import Config


def compare(config: Config,
            ac_checkpoint: str,
            reinforce_checkpoint: str,
            n_episodes: int = 100) -> None:
    """
    Evaluate both agents with the same random seeds.
    Prints a comparison table to stdout:

        Agent         | Mean Return | Std Return | Success Rate | Mean Steps
        --------------|-------------|------------|--------------|----------
        Actor-Critic  |     ...     |    ...     |     ...      |   ...
        REINFORCE     |     ...     |    ...     |     ...      |   ...

    Args:
        config:                Config matching both training runs.
        ac_checkpoint:         path to Actor-Critic checkpoint (.pt file).
        reinforce_checkpoint:  path to REINFORCE checkpoint (.pt file).
        n_episodes:            number of evaluation episodes per agent.
    """
    pass


if __name__ == "__main__":
    """
    Compare Actor-Critic and REINFORCE on identical evaluation episodes.
    Usage:
        python run/compare.py \
            --ac artifacts/checkpoints/ac_ep5000.pt \
            --reinforce artifacts/checkpoints/reinforce_ep5000.pt
    Args:
        --ac:          path to Actor-Critic checkpoint (required).
        --reinforce:   path to REINFORCE checkpoint (required).
        --n_episodes:  number of evaluation episodes (default: 100).
    Stdout: prints comparison table (see compare() docstring).
    Saves:  artifacts/logs/comparison.csv
            columns: agent, mean_return, std_return, success_rate, mean_steps.
    """
    pass
