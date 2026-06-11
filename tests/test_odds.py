"""Tests for odds conversion and margin removal."""

import pytest

from model.expected_goals import fit_lambdas, total_lambda_from_over_2_5
from model.odds import (
    implied_probabilities_1x2,
    implied_probabilities_over_under,
    market_probabilities,
)
from model.poisson import prob_total_over_poisson


class TestOddsConversion:
    def test_probabilities_sum_to_one(self):
        p_a, p_d, p_b, _ = implied_probabilities_1x2(2.10, 3.30, 3.60)
        assert p_a + p_d + p_b == pytest.approx(1.0)

    def test_overround_is_positive_for_real_odds(self):
        _, _, _, overround = implied_probabilities_1x2(2.10, 3.30, 3.60)
        assert overround > 0

    def test_favorite_has_highest_probability(self):
        p_a, p_d, p_b, _ = implied_probabilities_1x2(1.50, 4.00, 7.00)
        assert p_a > p_d > p_b

    def test_over_under_sums_to_one(self):
        p_over, p_under, _ = implied_probabilities_over_under(1.90, 1.90)
        assert p_over + p_under == pytest.approx(1.0)
        assert p_over == pytest.approx(0.5)

    def test_fair_odds_have_zero_margin(self):
        # Fair 3-way odds at exactly 1/3 probability each
        p_a, p_d, p_b, overround = implied_probabilities_1x2(3.0, 3.0, 3.0)
        assert overround == pytest.approx(0.0)
        assert p_a == pytest.approx(1 / 3)

    def test_invalid_odds_rejected(self):
        with pytest.raises(ValueError):
            implied_probabilities_1x2(0.95, 3.30, 3.60)
        with pytest.raises(ValueError):
            implied_probabilities_over_under(1.0, 2.0)

    def test_market_probabilities_bundle(self):
        m = market_probabilities(2.10, 3.30, 3.60, 1.90, 1.95)
        assert m.p_a_win + m.p_draw + m.p_b_win == pytest.approx(1.0)
        assert m.p_over_2_5 + m.p_under_2_5 == pytest.approx(1.0)


class TestExpectedGoals:
    def test_total_lambda_recovers_target_probability(self):
        lam = total_lambda_from_over_2_5(0.55)
        assert prob_total_over_poisson(lam) == pytest.approx(0.55, abs=1e-6)

    def test_total_lambda_respects_bounds(self):
        assert total_lambda_from_over_2_5(0.001) == pytest.approx(0.8)
        assert total_lambda_from_over_2_5(0.999) == pytest.approx(5.0)

    def test_fit_matches_market_closely(self):
        market = market_probabilities(2.10, 3.30, 3.60, 1.90, 1.95)
        est = fit_lambdas(market)
        # An independent-Poisson model has 2 parameters fitting 4 market
        # probabilities, so a small structural residual is expected (markets
        # price draws slightly above pure Poisson — the Dixon-Coles effect).
        assert est.fit_error < 5e-3
        assert est.lambda_a > est.lambda_b  # team A is the favorite
        # Each model outcome probability should be within 4pp of the market.
        from model.poisson import outcome_probabilities, score_probability_matrix

        p_a, p_d, p_b = outcome_probabilities(
            score_probability_matrix(est.lambda_a, est.lambda_b)
        )
        assert abs(p_a - market.p_a_win) < 0.04
        assert abs(p_d - market.p_draw) < 0.04
        assert abs(p_b - market.p_b_win) < 0.04

    def test_heavy_favorite_gets_much_higher_lambda(self):
        market = market_probabilities(1.20, 7.00, 15.00, 1.65, 2.25)
        est = fit_lambdas(market)
        assert est.lambda_a > 2.0 * est.lambda_b
