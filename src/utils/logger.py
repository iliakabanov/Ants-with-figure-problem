import csv


class Logger:
    """Logs per-episode metrics to a CSV file."""

    def __init__(self, log_path: str) -> None:
        """Args: log_path — path to output CSV."""
        pass

    def log_episode(self, episode: int, total_reward: float,
                    steps: int, success: bool,
                    losses: dict[str, float]) -> None:
        """Write one episode record to the CSV."""
        pass

    def close(self) -> None:
        """Flush and close the file."""
        pass
