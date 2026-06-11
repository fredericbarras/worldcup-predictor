"""Conversion of bookmaker decimal odds into fair (margin-free) probabilities.

Method: basic proportional normalization.
    implied = 1 / decimal_odds
    fair    = implied / sum(implied over the market)

This removes the bookmaker overround proportionally across outcomes,
which is the standard simple approach and adequate for this use case.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketProbabilities:
    """Fair (margin-removed) market probabilities for one match."""

    p_a_win: float
    p_draw: float
    p_b_win: float
    p_over_2_5: float
    p_under_2_5: float
    overround_1x2: float  # bookmaker margin on the 1X2 market, e.g. 0.05 = 5%
    overround_ou: float   # bookmaker margin on the over/under market


def implied_probabilities_1x2(odds_a: float, odds_draw: float, odds_b: float):
    """Return (p_a, p_draw, p_b, overround) with margin removed."""
    _check_odds(odds_a, odds_draw, odds_b)
    raw_a = 1.0 / odds_a
    raw_d = 1.0 / odds_draw
    raw_b = 1.0 / odds_b
    total = raw_a + raw_d + raw_b
    return raw_a / total, raw_d / total, raw_b / total, total - 1.0


def implied_probabilities_over_under(odds_over: float, odds_under: float):
    """Return (p_over, p_under, overround) with margin removed."""
    _check_odds(odds_over, odds_under)
    raw_o = 1.0 / odds_over
    raw_u = 1.0 / odds_under
    total = raw_o + raw_u
    return raw_o / total, raw_u / total, total - 1.0


def market_probabilities(
    odds_a: float,
    odds_draw: float,
    odds_b: float,
    odds_over_2_5: float,
    odds_under_2_5: float,
) -> MarketProbabilities:
    """Convert the full set of input odds into fair probabilities."""
    p_a, p_d, p_b, over_1x2 = implied_probabilities_1x2(odds_a, odds_draw, odds_b)
    p_over, p_under, over_ou = implied_probabilities_over_under(
        odds_over_2_5, odds_under_2_5
    )
    return MarketProbabilities(
        p_a_win=p_a,
        p_draw=p_d,
        p_b_win=p_b,
        p_over_2_5=p_over,
        p_under_2_5=p_under,
        overround_1x2=over_1x2,
        overround_ou=over_ou,
    )


def _check_odds(*odds: float) -> None:
    for o in odds:
        if o is None or o <= 1.0:
            raise ValueError(
                f"Decimal odds must be greater than 1.0, got {o!r}."
            )
