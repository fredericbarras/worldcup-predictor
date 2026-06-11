"""Tests for the Monte Carlo simulation and expected-points optimizer."""

import numpy as np
import pytest

from model.monte_carlo import simulate_match
from model.optimizer import (
    best_prediction,
    evaluate_all_predictions,
    most_likely_score,
)
from model.scoring import score_prediction_points_vectorized


@pytest.fixture(scope="module")
def sim():
    return simulate_match(1.8, 1.0, n_simulations=30_000, seed=7)


class TestSimulation:
    def test_probabilities_sum_to_one(self, sim):
        assert sim.p_a_win + sim.p_draw + sim.p_b_win == pytest.approx(1.0)

    def test_means_close_to_lambdas(self, sim):
        assert sim.mean_goals_a == pytest.approx(1.8, abs=0.05)
        assert sim.mean_goals_b == pytest.approx(1.0, abs=0.05)

    def test_reproducible_with_seed(self):
        a = simulate_match(1.5, 1.2, n_simulations=5_000, seed=123)
        b = simulate_match(1.5, 1.2, n_simulations=5_000, seed=123)
        assert np.array_equal(a.goals_a, b.goals_a)
        assert np.array_equal(a.goals_b, b.goals_b)


class TestOptimizer:
    def test_returns_valid_scorelines(self, sim):
        df = evaluate_all_predictions(sim, "Group Stage", max_pred_goals=6)
        assert len(df) == 49
        assert df["pred_a"].between(0, 6).all()
        assert df["pred_b"].between(0, 6).all()

    def test_expected_points_are_finite_and_bounded(self, sim):
        df = evaluate_all_predictions(sim, "Group Stage")
        assert df["expected_points"].between(0, 9).all()  # max 5+1+3
        df_ko = evaluate_all_predictions(sim, "Final")
        assert df_ko["expected_points"].between(0, 18).all()  # max 10+2+6

    def test_recommendation_has_highest_expected_points(self, sim):
        df = evaluate_all_predictions(sim, "Group Stage")
        best = best_prediction(df)
        assert best["expected_points"] == df["expected_points"].max()

    def test_expected_points_match_direct_computation(self, sim):
        df = evaluate_all_predictions(sim, "Round of 16")
        best = best_prediction(df)
        direct = score_prediction_points_vectorized(
            int(best["pred_a"]),
            int(best["pred_b"]),
            sim.goals_a,
            sim.goals_b,
            "Round of 16",
        ).mean()
        assert best["expected_points"] == pytest.approx(direct)

    def test_favorite_recommended_to_win(self, sim):
        # With lambdas 1.8 vs 1.0, the optimizer should recommend a team A win.
        best = best_prediction(evaluate_all_predictions(sim, "Group Stage"))
        assert best["pred_a"] > best["pred_b"]

    def test_most_likely_score_maximizes_exact_probability(self, sim):
        df = evaluate_all_predictions(sim, "Group Stage")
        ml = most_likely_score(df)
        assert ml["p_exact"] == df["p_exact"].max()
