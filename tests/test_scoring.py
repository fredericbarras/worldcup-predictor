"""Tests for the family competition scoring rules."""

import numpy as np

from model.scoring import (
    score_prediction_points,
    score_prediction_points_vectorized,
)


class TestGroupStage:
    def test_correct_winner_and_goal_difference(self):
        # Predicted France 2-1, actual France 1-0:
        # winner correct (5) + total wrong (0) + diff correct (3) = 8
        assert score_prediction_points(2, 1, 1, 0, "Group Stage") == 8

    def test_correct_winner_only(self):
        # Predicted France 2-1, actual France 3-1:
        # winner correct (5) + total wrong (0) + diff wrong (0) = 5
        assert score_prediction_points(2, 1, 3, 1, "Group Stage") == 5

    def test_draw_no_goal_difference_bonus(self):
        # Predicted 1-1, actual 0-0:
        # draw correct (5) + total wrong (0) + no diff bonus for draws = 5
        assert score_prediction_points(1, 1, 0, 0, "Group Stage") == 5

    def test_exact_score(self):
        # Exact 2-1: winner (5) + total (1) + diff (3) = 9
        assert score_prediction_points(2, 1, 2, 1, "Group Stage") == 9

    def test_exact_draw(self):
        # Exact 1-1: draw (5) + total (1), no diff bonus for draws = 6
        assert score_prediction_points(1, 1, 1, 1, "Group Stage") == 6

    def test_wrong_winner_correct_total(self):
        # Predicted 2-1 (A win), actual 1-2 (B win): only total goals (1)
        assert score_prediction_points(2, 1, 1, 2, "Group Stage") == 1

    def test_everything_wrong(self):
        assert score_prediction_points(2, 0, 0, 1, "Group Stage") == 0


class TestKnockout:
    def test_correct_winner_and_goal_difference(self):
        # Predicted Argentina 2-1, actual Argentina 1-0:
        # winner (10) + total wrong (0) + diff correct (6) = 16
        assert score_prediction_points(2, 1, 1, 0, "Round of 16") == 16

    def test_draw_after_120_minutes(self):
        # Predicted 1-1, actual 2-2 after 120 minutes:
        # draw correct (10), total wrong, no diff bonus for draws = 10
        assert score_prediction_points(1, 1, 2, 2, "Quarterfinal") == 10

    def test_exact_knockout_score(self):
        # Exact 2-1 in the final: 10 + 2 + 6 = 18
        assert score_prediction_points(2, 1, 2, 1, "Final") == 18

    def test_all_knockout_phases_use_same_weights(self):
        for phase in (
            "Round of 32",
            "Round of 16",
            "Quarterfinal",
            "Semifinal",
            "Third-place match",
            "Final",
        ):
            assert score_prediction_points(2, 1, 1, 0, phase) == 16


class TestVectorizedConsistency:
    def test_matches_scalar_version(self):
        rng = np.random.default_rng(42)
        actual_a = rng.poisson(1.5, size=500)
        actual_b = rng.poisson(1.1, size=500)
        for phase in ("Group Stage", "Final"):
            for pa, pb in [(0, 0), (1, 1), (2, 1), (1, 2), (3, 0)]:
                vec = score_prediction_points_vectorized(
                    pa, pb, actual_a, actual_b, phase
                )
                scalar = [
                    score_prediction_points(pa, pb, int(a), int(b), phase)
                    for a, b in zip(actual_a, actual_b)
                ]
                assert vec.tolist() == scalar
