"""Local CSV storage for saved predictions and reference data."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.csv"

PREDICTION_COLUMNS = [
    "timestamp",
    "team_a",
    "team_b",
    "phase",
    "kickoff_time",
    "odds_a",
    "odds_draw",
    "odds_b",
    "odds_over_2_5",
    "odds_under_2_5",
    "lambda_a_baseline",
    "lambda_b_baseline",
    "lambda_a_adjusted",
    "lambda_b_adjusted",
    "simulations",
    "recommended_pred_a",
    "recommended_pred_b",
    "expected_points",
    "most_likely_score_a",
    "most_likely_score_b",
    "most_likely_score_probability",
    "sim_p_a_win",
    "sim_p_draw",
    "sim_p_b_win",
    "notes",
]


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists() or PREDICTIONS_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.read_csv(PREDICTIONS_FILE)


def append_prediction(record: dict) -> None:
    """Append one prediction row; unknown keys are ignored, missing keys blank."""
    row = {col: record.get(col, "") for col in PREDICTION_COLUMNS}
    df = load_predictions()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_FILE, index=False)


def load_teams() -> pd.DataFrame:
    path = DATA_DIR / "teams.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["team_id", "team_name"])


def load_fixtures() -> pd.DataFrame:
    path = DATA_DIR / "fixtures.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()
