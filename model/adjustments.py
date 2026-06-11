"""Transparent, bounded adjustments applied on top of the market-implied lambdas.

Design principles
-----------------
* The betting market is the baseline. Adjustments are small additive xG nudges.
* Every adjustment is numeric, bounded, and documented below — no opaque sliders.
* Adjustments are applied AFTER the baseline fit:
      lambda_adjusted = max(0.05, lambda_baseline + sum(adjustments))

Adjustment semantics (all values are in expected goals, xG):

  recent_form_a / recent_form_b        [-0.15, +0.15]
      Sustained over/under-performance in the last ~5 matches.
  injury_a / injury_b                  [-0.30, +0.30]
      Impact of missing key players (negative if weakened).
  lineup_a / lineup_b                  [-0.30, +0.30]
      Confirmed lineup strength vs. expected (e.g. star player rested).
  rest_a / rest_b                      [-0.10, +0.10]
      Rest days / travel / fatigue differential.
  motivation_a / motivation_b          [-0.15, +0.15]
      Context: already qualified, must-win, dead rubber, derby.
  host_advantage                       [0.0, +0.25], applied to ONE team
      Meaningful home/host crowd effect (USA/Mexico/Canada in 2026).

Rating-based components (converted to xG with small documented coefficients):

  elo_diff (team A Elo minus team B Elo)
      xG effect = clip(elo_diff * ELO_TO_XG, -0.25, +0.25), split symmetrically:
      +half to team A, -half to team B. ELO_TO_XG = 0.0008 means a 100-point
      Elo gap shifts the xG balance by 0.08 goals total — deliberately small,
      because the market already prices most of the rating gap.
  fifa_rank_diff (team B rank minus team A rank, positive = A better ranked)
      xG effect = clip(fifa_rank_diff * RANK_TO_XG, -0.10, +0.10), split
      symmetrically. RANK_TO_XG = 0.002 (a 25-place gap ≈ 0.05 goals total).

If you trust the market fully, leave everything at 0.
"""

from dataclasses import dataclass, field

ELO_TO_XG = 0.0008
ELO_EFFECT_CAP = 0.25
RANK_TO_XG = 0.002
RANK_EFFECT_CAP = 0.10
MIN_LAMBDA = 0.05


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class AdjustmentInputs:
    """All optional manual adjustments. Defaults are neutral (zero effect)."""

    elo_diff: float = 0.0          # team A Elo - team B Elo
    fifa_rank_diff: float = 0.0    # team B FIFA rank - team A FIFA rank
    recent_form_a: float = 0.0
    recent_form_b: float = 0.0
    injury_a: float = 0.0
    injury_b: float = 0.0
    lineup_a: float = 0.0
    lineup_b: float = 0.0
    rest_a: float = 0.0
    rest_b: float = 0.0
    motivation_a: float = 0.0
    motivation_b: float = 0.0
    host_advantage: float = 0.0    # magnitude of the host effect in xG
    host_team: str = "none"        # "none" | "a" | "b"
    notes: str = field(default="")


def compute_adjustments(inputs: AdjustmentInputs):
    """Return (adjustment_a, adjustment_b) in xG, fully transparent and additive."""
    elo_effect = _clip(inputs.elo_diff * ELO_TO_XG, -ELO_EFFECT_CAP, ELO_EFFECT_CAP)
    rank_effect = _clip(
        inputs.fifa_rank_diff * RANK_TO_XG, -RANK_EFFECT_CAP, RANK_EFFECT_CAP
    )

    adj_a = (
        inputs.recent_form_a
        + inputs.injury_a
        + inputs.lineup_a
        + inputs.rest_a
        + inputs.motivation_a
        + elo_effect / 2.0
        + rank_effect / 2.0
    )
    adj_b = (
        inputs.recent_form_b
        + inputs.injury_b
        + inputs.lineup_b
        + inputs.rest_b
        + inputs.motivation_b
        - elo_effect / 2.0
        - rank_effect / 2.0
    )

    if inputs.host_team == "a":
        adj_a += inputs.host_advantage
    elif inputs.host_team == "b":
        adj_b += inputs.host_advantage

    return adj_a, adj_b


def apply_adjustments(lambda_a: float, lambda_b: float, inputs: AdjustmentInputs):
    """Apply adjustments to baseline lambdas; floor at MIN_LAMBDA."""
    adj_a, adj_b = compute_adjustments(inputs)
    return (
        max(MIN_LAMBDA, lambda_a + adj_a),
        max(MIN_LAMBDA, lambda_b + adj_b),
    )
