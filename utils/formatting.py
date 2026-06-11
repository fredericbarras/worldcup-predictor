"""Display formatting helpers."""


def pct(value: float, decimals: int = 1) -> str:
    """0.1234 -> '12.3%'"""
    return f"{value * 100:.{decimals}f}%"


def score_label(goals_a: int, goals_b: int) -> str:
    return f"{int(goals_a)}-{int(goals_b)}"


def signed(value: float, decimals: int = 2) -> str:
    """0.15 -> '+0.15'"""
    return f"{value:+.{decimals}f}"
