"""Monte Carlo simulation of match scorelines from Poisson lambdas."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    goals_a: np.ndarray
    goals_b: np.ndarray

    @property
    def n(self) -> int:
        return len(self.goals_a)

    @property
    def p_a_win(self) -> float:
        return float(np.mean(self.goals_a > self.goals_b))

    @property
    def p_draw(self) -> float:
        return float(np.mean(self.goals_a == self.goals_b))

    @property
    def p_b_win(self) -> float:
        return float(np.mean(self.goals_a < self.goals_b))

    @property
    def mean_goals_a(self) -> float:
        return float(np.mean(self.goals_a))

    @property
    def mean_goals_b(self) -> float:
        return float(np.mean(self.goals_b))

    def prob_over(self, line: float = 2.5) -> float:
        return float(np.mean((self.goals_a + self.goals_b) > line))

    def prob_btts(self) -> float:
        """Both teams to score."""
        return float(np.mean((self.goals_a > 0) & (self.goals_b > 0)))

    def score_probability_table(self, max_goals: int = 6) -> pd.DataFrame:
        """Probability matrix; goals above max_goals are pooled into the last bin."""
        a = np.minimum(self.goals_a, max_goals)
        b = np.minimum(self.goals_b, max_goals)
        counts = np.zeros((max_goals + 1, max_goals + 1), dtype=np.int64)
        np.add.at(counts, (a, b), 1)
        probs = counts / self.n
        labels = [str(i) for i in range(max_goals)] + [f"{max_goals}+"]
        return pd.DataFrame(probs, index=labels, columns=labels)

    def top_scores(self, k: int = 10) -> pd.DataFrame:
        """Most frequent exact scorelines, descending."""
        df = pd.DataFrame({"goals_a": self.goals_a, "goals_b": self.goals_b})
        top = (
            df.value_counts(normalize=True)
            .head(k)
            .rename("probability")
            .reset_index()
        )
        top["score"] = top["goals_a"].astype(str) + "-" + top["goals_b"].astype(str)
        return top[["score", "goals_a", "goals_b", "probability"]]

    def to_dataframe(self) -> pd.DataFrame:
        diff = self.goals_a - self.goals_b
        winner = np.where(diff > 0, "A", np.where(diff < 0, "B", "draw"))
        return pd.DataFrame(
            {
                "goals_A": self.goals_a,
                "goals_B": self.goals_b,
                "winner": winner,
                "total_goals": self.goals_a + self.goals_b,
                "goal_difference": diff,
            }
        )


def simulate_match(
    lambda_a: float,
    lambda_b: float,
    n_simulations: int = 50_000,
    seed: int | None = None,
) -> SimulationResult:
    """Draw n_simulations independent Poisson scorelines."""
    if lambda_a <= 0 or lambda_b <= 0:
        raise ValueError("Lambdas must be positive.")
    if n_simulations < 1000:
        raise ValueError("Use at least 1,000 simulations for stable estimates.")
    rng = np.random.default_rng(seed)
    goals_a = rng.poisson(lambda_a, size=n_simulations)
    goals_b = rng.poisson(lambda_b, size=n_simulations)
    return SimulationResult(goals_a=goals_a, goals_b=goals_b)
