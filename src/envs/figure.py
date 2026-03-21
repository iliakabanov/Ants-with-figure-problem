import pymunk


class TFigure:
    """
    Rigid T-shaped figure: top bar (5x1), leg (4x1), bottom bar (2x1).
    Wraps a pymunk Body with three Shape objects.
    """

    def __init__(self, space: pymunk.Space) -> None:
        """Create pymunk body and attach three rectangular shapes."""
        pass

    def set_state(self, x: float, y: float, theta: float) -> None:
        """Set figure position and orientation."""
        pass

    def get_corners(self) -> list[tuple[float, float]]:
        """Return world-space coordinates of all figure corners."""
        pass

    def compute_progress(self, wall_geometry: dict) -> float:
        """
        Compute rho_i(s) = S_i_wall(s) / S_total
        for a given wall pair described by wall_geometry:
            { 'x': float, 'y_gap': float, 'gap_width': float }
        Only the area of the figure to the right of wall_x
        is counted as S_i_wall.
        """
        pass

    def get_total_area(self) -> float:
        """Return total area S_total of the T-figure."""
        pass
