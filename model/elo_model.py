"""Elo-based expected-goals fallback — used when no betting odds are available.

This lets the app produce a recommendation for ANY match with zero manual
input. It is deliberately simple and transparent, and less sharp than the
market model: when odds are available (manual or fetched), they take priority.

Method
------
1. Elo win expectancy:    E_A = 1 / (1 + 10^(-(elo_A - elo_B) / 400))
2. Expected goal margin:  diff = clip(elo_diff / 220, -3, +3)
   (≈ 0.45 goals per 100 Elo points — a standard football-Elo heuristic.)
3. Expected total goals:  total = clip(2.5 + 0.25 * |diff|, 0.8, 4.5)
   (mismatches produce more goals; 2.5 is the World Cup baseline.)
4. Split:                 lambda_A = (total + diff) / 2, floor at 0.15.

Elo ratings are read from data/elo_ratings.csv — update that file from
eloratings.net before the tournament for best results.
"""

from pathlib import Path

import pandas as pd

ELO_FILE = Path(__file__).resolve().parent.parent / "data" / "elo_ratings.csv"

GOALS_PER_ELO = 1.0 / 220.0   # expected goal margin per Elo point
MAX_GOAL_DIFF = 3.0
BASE_TOTAL_GOALS = 2.5
TOTAL_PER_DIFF = 0.25
MIN_LAMBDA = 0.15
DEFAULT_ELO = 1700.0          # used when a team is missing from the ratings file


def load_elo_ratings() -> dict[str, float]:
    if not ELO_FILE.exists():
        return {}
    df = pd.read_csv(ELO_FILE)
    return dict(zip(df["team_name"], df["elo_rating"].astype(float)))


def win_expectancy(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 400.0))


def lambdas_from_elo(elo_a: float, elo_b: float) -> tuple[float, float]:
    """Expected goals for both teams from Elo ratings alone."""
    diff = max(-MAX_GOAL_DIFF, min(MAX_GOAL_DIFF, (elo_a - elo_b) * GOALS_PER_ELO))
    total = min(4.5, max(0.8, BASE_TOTAL_GOALS + TOTAL_PER_DIFF * abs(diff)))
    lam_a = max(MIN_LAMBDA, (total + diff) / 2.0)
    lam_b = max(MIN_LAMBDA, (total - diff) / 2.0)
    return lam_a, lam_b


def lambdas_for_teams(
    team_a: str, team_b: str, ratings: dict[str, float] | None = None
) -> tuple[float, float, float, float]:
    """Convenience wrapper: returns (lambda_a, lambda_b, elo_a, elo_b)."""
    ratings = ratings if ratings is not None else load_elo_ratings()
    elo_a = ratings.get(team_a, DEFAULT_ELO)
    elo_b = ratings.get(team_b, DEFAULT_ELO)
    lam_a, lam_b = lambdas_from_elo(elo_a, elo_b)
    return lam_a, lam_b, elo_a, elo_b
