"""Poisson scoreline model helpers.

The match model assumes independent Poisson goal counts:
    goals_A ~ Poisson(lambda_A)
    goals_B ~ Poisson(lambda_B)
"""

import numpy as np
from scipy import stats

# Goals are truncated at this value when building probability grids.
# P(goals > 15) is negligible for any realistic lambda (< 1e-9 at lambda=5).
GRID_MAX_GOALS = 15


def poisson_pmf_vector(lam: float, max_goals: int = GRID_MAX_GOALS) -> np.ndarray:
    """P(goals = k) for k in 0..max_goals."""
    return stats.poisson.pmf(np.arange(max_goals + 1), lam)


def score_probability_matrix(
    lambda_a: float, lambda_b: float, max_goals: int = GRID_MAX_GOALS
) -> np.ndarray:
    """Joint probability matrix M[i, j] = P(team A scores i, team B scores j)."""
    pa = poisson_pmf_vector(lambda_a, max_goals)
    pb = poisson_pmf_vector(lambda_b, max_goals)
    return np.outer(pa, pb)


def outcome_probabilities(matrix: np.ndarray):
    """Return (p_a_win, p_draw, p_b_win) from a score probability matrix."""
    p_a_win = float(np.tril(matrix, k=-1).sum())  # rows (A goals) > cols (B goals)
    p_draw = float(np.trace(matrix))
    p_b_win = float(np.triu(matrix, k=1).sum())
    return p_a_win, p_draw, p_b_win


def prob_total_over(matrix: np.ndarray, line: float = 2.5) -> float:
    """P(total goals > line) from a score probability matrix."""
    n = matrix.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    return float(matrix[totals > line].sum())


def prob_total_over_poisson(total_lambda: float, line: float = 2.5) -> float:
    """P(N > line) where N ~ Poisson(total_lambda)."""
    return float(1.0 - stats.poisson.cdf(int(np.floor(line)), total_lambda))
