from src.utils.config import Config


def record_rollout(config: Config, checkpoint_path: str,
                   output_path: str) -> None:
    """
    Run one deterministic episode and save frames as a GIF/MP4.

    Args:
        config:          Config matching the training run.
        checkpoint_path: path to saved model checkpoint.
        output_path:     destination path for the output file.
    """
    pass


if __name__ == "__main__":
    """
    Record a single rollout of the trained policy as a GIF or MP4.
    Usage: python run/record.py --checkpoint artifacts/checkpoints/ac_ep5000.pt
                                --output artifacts/videos/rollout.gif
    """
    pass
