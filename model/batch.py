"""Batch simulation: recommendations for every upcoming match in one pass.

For each fixture with both teams decided and no result yet:
  1. Use fetched market odds if available (best),
  2. otherwise fall back to the Elo model (no input required).
Then fit lambdas, simulate, and pick the expected-points-maximizing score.
"""

import pandas as pd

from api.odds_api import match_odds_to_fixture
from model.elo_model import lambdas_for_teams, load_elo_ratings
from model.expected_goals import fit_lambdas
from model.monte_carlo import simulate_match
from model.odds import market_probabilities
from model.optimizer import best_prediction, evaluate_all_predictions, most_likely_score


def recommend_for_match(
    team_a: str,
    team_b: str,
    phase: str,
    odds: dict | None = None,
    elo_ratings: dict | None = None,
    n_simulations: int = 20_000,
    max_pred_goals: int = 6,
    seed: int | None = None,
) -> dict:
    """Run the full pipeline for one match and return a summary row."""
    if odds is not None:
        market = market_probabilities(
            odds["odds_a"],
            odds["odds_draw"],
            odds["odds_b"],
            odds["odds_over_2_5"],
            odds["odds_under_2_5"],
        )
        estimate = fit_lambdas(market)
        lam_a, lam_b = estimate.lambda_a, estimate.lambda_b
        source = "market odds"
    else:
        lam_a, lam_b, _, _ = lambdas_for_teams(team_a, team_b, elo_ratings)
        source = "Elo fallback"

    sim = simulate_match(lam_a, lam_b, n_simulations, seed)
    evaluation = evaluate_all_predictions(sim, phase, max_pred_goals)
    best = best_prediction(evaluation)
    likely = most_likely_score(evaluation)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "phase": phase,
        "source": source,
        "lambda_a": round(lam_a, 3),
        "lambda_b": round(lam_b, 3),
        "recommended": best["score"],
        "recommended_pred_a": int(best["pred_a"]),
        "recommended_pred_b": int(best["pred_b"]),
        "expected_points": round(float(best["expected_points"]), 3),
        "most_likely": likely["score"],
        "most_likely_prob": round(float(likely["p_exact"]), 4),
        "p_a_win": round(sim.p_a_win, 3),
        "p_draw": round(sim.p_draw, 3),
        "p_b_win": round(sim.p_b_win, 3),
        "odds_a": odds["odds_a"] if odds else None,
        "odds_draw": odds["odds_draw"] if odds else None,
        "odds_b": odds["odds_b"] if odds else None,
        "odds_over_2_5": odds["odds_over_2_5"] if odds else None,
        "odds_under_2_5": odds["odds_under_2_5"] if odds else None,
    }


def simulate_all_upcoming(
    fixtures: pd.DataFrame,
    odds_events: list[dict] | None = None,
    n_simulations: int = 20_000,
    seed: int | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Recommendations for every fixture in the given (upcoming) fixtures frame.

    odds_events: output of api.odds_api.fetch_all_match_odds, or None to use
    the Elo fallback for every match.
    """
    elo_ratings = load_elo_ratings()
    rows = []
    total = len(fixtures)
    for i, fx in enumerate(fixtures.itertuples(index=False)):
        odds = (
            match_odds_to_fixture(odds_events, fx.team_a, fx.team_b)
            if odds_events
            else None
        )
        row = recommend_for_match(
            fx.team_a,
            fx.team_b,
            fx.phase,
            odds=odds,
            elo_ratings=elo_ratings,
            n_simulations=n_simulations,
            seed=seed,
        )
        row = {
            "match_id": fx.match_id,
            "date": fx.date,
            "kickoff_utc": fx.kickoff_utc,
            "group": fx.group,
            **row,
        }
        rows.append(row)
        if progress_callback:
            progress_callback((i + 1) / total, f"{fx.team_a} vs {fx.team_b}")
    return pd.DataFrame(rows)
