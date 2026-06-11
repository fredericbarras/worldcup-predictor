"""Expected-points optimization over all candidate scoreline predictions.

For every candidate prediction (0-0 up to max_pred_goals-max_pred_goals),
score it against every Monte Carlo simulation under the family rules and
average. The recommendation is the candidate with the highest expected
points — which is often NOT the most likely exact score, because a good
prediction also harvests winner and goal-difference points across many
neighboring scorelines.
"""

import numpy as np
import pandas as pd

from model.monte_carlo import SimulationResult
from model.scoring import score_prediction_points_vectorized


def evaluate_all_predictions(
    sim: SimulationResult, phase: str, max_pred_goals: int = 6
) -> pd.DataFrame:
    """Expected points for every candidate prediction, sorted descending.

    Columns:
        pred_a, pred_b, score, expected_points,
        p_exact, p_winner_correct, p_total_correct, p_diff_correct
    """
    actual_a, actual_b = sim.goals_a, sim.goals_b
    actual_sign = np.sign(actual_a.astype(np.int64) - actual_b.astype(np.int64))
    n = sim.n

    rows = []
    for pa in range(max_pred_goals + 1):
        for pb in range(max_pred_goals + 1):
            points = score_prediction_points_vectorized(
                pa, pb, actual_a, actual_b, phase
            )
            winner_correct = actual_sign == np.sign(pa - pb)
            rows.append(
                {
                    "pred_a": pa,
                    "pred_b": pb,
                    "score": f"{pa}-{pb}",
                    "expected_points": float(points.mean()),
                    "p_exact": float(
                        np.mean((actual_a == pa) & (actual_b == pb))
                    ),
                    "p_winner_correct": float(winner_correct.mean()),
                    "p_total_correct": float(
                        np.mean((actual_a + actual_b) == (pa + pb))
                    ),
                    "p_diff_correct": float(
                        np.mean((actual_a - actual_b) == (pa - pb))
                    ),
                }
            )

    df = pd.DataFrame(rows).sort_values(
        "expected_points", ascending=False, ignore_index=True
    )
    assert len(df) == (max_pred_goals + 1) ** 2 and n > 0
    return df


def best_prediction(evaluation: pd.DataFrame) -> pd.Series:
    """The candidate with the highest expected points."""
    return evaluation.iloc[0]


def most_likely_score(evaluation: pd.DataFrame) -> pd.Series:
    """The candidate with the highest exact-score probability."""
    return evaluation.loc[evaluation["p_exact"].idxmax()]
