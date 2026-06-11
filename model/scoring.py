"""Family competition scoring rules.

Group stage:
    5 points  — correct winner (or correctly predicted draw)
    1 point   — correct total number of goals
    3 points  — correct goal difference, ONLY if the winner is also correct

All knockout rounds (R32, R16, QF, SF, third place, final):
    10 / 2 / 6 points with the same logic.
    The result after 120 minutes counts; penalty shootout goals are excluded,
    so a knockout match CAN end in a predicted draw.

EXPLICIT ASSUMPTION (easy to change):
    The goal-difference bonus is only awarded for non-draw outcomes, because
    the rule says "le gagnant doit être correct" and a draw has no winner.
    A correctly predicted draw earns only the winner points (plus total-goals
    points if applicable). To change this, set DRAW_EARNS_DIFF_BONUS = True.
"""

from dataclasses import dataclass

import numpy as np

DRAW_EARNS_DIFF_BONUS = False

GROUP_STAGE = "Group Stage"
KNOCKOUT_PHASES = (
    "Round of 32",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Third-place match",
    "Final",
)
ALL_PHASES = (GROUP_STAGE,) + KNOCKOUT_PHASES


@dataclass(frozen=True)
class ScoringWeights:
    winner_points: int
    total_goals_points: int
    goal_diff_points: int

    @property
    def max_points(self) -> int:
        return self.winner_points + self.total_goals_points + self.goal_diff_points


def get_weights(phase: str) -> ScoringWeights:
    if phase == GROUP_STAGE:
        return ScoringWeights(winner_points=5, total_goals_points=1, goal_diff_points=3)
    if phase in KNOCKOUT_PHASES:
        return ScoringWeights(winner_points=10, total_goals_points=2, goal_diff_points=6)
    raise ValueError(f"Unknown phase: {phase!r}. Expected one of {ALL_PHASES}.")


def score_prediction_points(
    pred_a: int, pred_b: int, actual_a: int, actual_b: int, phase: str
) -> int:
    """Points earned by predicting pred_a-pred_b when the result is actual_a-actual_b."""
    w = get_weights(phase)
    points = 0

    pred_sign = np.sign(pred_a - pred_b)
    actual_sign = np.sign(actual_a - actual_b)
    winner_correct = pred_sign == actual_sign

    if winner_correct:
        points += w.winner_points

    if (pred_a + pred_b) == (actual_a + actual_b):
        points += w.total_goals_points

    diff_correct = (pred_a - pred_b) == (actual_a - actual_b)
    is_draw = actual_sign == 0
    if diff_correct and winner_correct and (DRAW_EARNS_DIFF_BONUS or not is_draw):
        points += w.goal_diff_points

    return int(points)


def score_prediction_points_vectorized(
    pred_a: int,
    pred_b: int,
    actual_a: np.ndarray,
    actual_b: np.ndarray,
    phase: str,
) -> np.ndarray:
    """Same rules as score_prediction_points, applied to arrays of actual results.

    Used by the optimizer to score one candidate prediction against every
    Monte Carlo simulation at once.
    """
    w = get_weights(phase)

    pred_sign = np.sign(pred_a - pred_b)
    actual_sign = np.sign(actual_a - actual_b)
    winner_correct = actual_sign == pred_sign

    total_correct = (actual_a + actual_b) == (pred_a + pred_b)

    diff_correct = (actual_a - actual_b) == (pred_a - pred_b)
    is_draw = actual_sign == 0
    diff_bonus = diff_correct & winner_correct
    if not DRAW_EARNS_DIFF_BONUS:
        diff_bonus &= ~is_draw

    return (
        winner_correct * w.winner_points
        + total_correct * w.total_goals_points
        + diff_bonus * w.goal_diff_points
    )
