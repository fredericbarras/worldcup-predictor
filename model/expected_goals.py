"""Estimate expected goals (lambdas) for both teams from market probabilities.

Two-step procedure:

1. Infer total expected goals from the Over/Under 2.5 market by root-finding:
   find total_lambda such that P(total > 2.5 | Poisson(total_lambda)) = p_over.

2. Split the total between the two teams by least-squares optimization:
   search (lambda_A, lambda_B) so that the implied Poisson model best matches
   the four market probabilities (A win, draw, B win, over 2.5).
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from model.odds import MarketProbabilities
from model.poisson import (
    outcome_probabilities,
    prob_total_over,
    prob_total_over_poisson,
    score_probability_matrix,
)

TOTAL_LAMBDA_BOUNDS = (0.8, 5.0)
TEAM_LAMBDA_BOUNDS = (0.1, 5.0)


@dataclass(frozen=True)
class LambdaEstimate:
    lambda_a: float
    lambda_b: float
    total_lambda: float
    fit_error: float  # residual sum of squared errors vs. market probabilities


def total_lambda_from_over_2_5(p_over: float) -> float:
    """Root-find total expected goals from P(total goals > 2.5)."""
    lo, hi = TOTAL_LAMBDA_BOUNDS
    # Clamp the target into the range achievable within the bounds so that
    # extreme odds still produce a sensible (boundary) estimate.
    p_lo = prob_total_over_poisson(lo)
    p_hi = prob_total_over_poisson(hi)
    if p_over <= p_lo:
        return lo
    if p_over >= p_hi:
        return hi
    return float(
        optimize.brentq(lambda lam: prob_total_over_poisson(lam) - p_over, lo, hi)
    )


def _initial_split(total_lambda: float, p_a_win: float, p_b_win: float):
    """Heuristic starting point: split the total by the win-probability ratio."""
    ratio = np.sqrt(max(p_a_win, 1e-6) / max(p_b_win, 1e-6))
    lam_a = total_lambda * ratio / (1.0 + ratio)
    lam_b = total_lambda - lam_a
    lo, hi = TEAM_LAMBDA_BOUNDS
    return float(np.clip(lam_a, lo, hi)), float(np.clip(lam_b, lo, hi))


def fit_lambdas(market: MarketProbabilities) -> LambdaEstimate:
    """Find (lambda_A, lambda_B) that best match the market probabilities."""
    total_lambda = total_lambda_from_over_2_5(market.p_over_2_5)
    x0 = _initial_split(total_lambda, market.p_a_win, market.p_b_win)

    target = np.array(
        [market.p_a_win, market.p_draw, market.p_b_win, market.p_over_2_5]
    )

    def objective(x: np.ndarray) -> float:
        matrix = score_probability_matrix(x[0], x[1])
        p_a, p_d, p_b = outcome_probabilities(matrix)
        p_over = prob_total_over(matrix, 2.5)
        model = np.array([p_a, p_d, p_b, p_over])
        return float(np.sum((model - target) ** 2))

    result = optimize.minimize(
        objective,
        x0=np.array(x0),
        method="L-BFGS-B",
        bounds=[TEAM_LAMBDA_BOUNDS, TEAM_LAMBDA_BOUNDS],
    )
    lam_a, lam_b = result.x
    return LambdaEstimate(
        lambda_a=float(lam_a),
        lambda_b=float(lam_b),
        total_lambda=total_lambda,
        fit_error=float(result.fun),
    )
