from src.utils.config import Config


def build_agent(config: Config,
                agent_name: str,
                baseline_name: str):
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
    pass


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
    pass


if __name__ == "__main__":
    """
    Train a selected agent from scratch.
    Usage:
        python run/train.py --agent ac
        python run/train.py --agent reinforce
        python run/train.py --agent reinforce --baseline mean_return
        python run/train.py --agent ac --seed 123
    Args:
        --agent:    'ac' or 'reinforce' (required).
        --baseline: 'zero' or 'mean_return' (default: 'zero').
        --seed:     integer random seed (default: Config.seed).
    Saves checkpoints to: artifacts/checkpoints/{agent}_ep{N}.pt
    Saves log to:         artifacts/logs/{agent}.csv
                          columns: episode, return, steps, success
    Stdout: prints eval mean return and success rate every eval_every episodes.
    """
    pass
