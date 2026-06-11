"""Input validation helpers for the Streamlit app."""


def validate_odds(*odds: float) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems = []
    labels = ["Team A win", "Draw", "Team B win", "Over 2.5", "Under 2.5"]
    for label, o in zip(labels, odds):
        if o is None or o <= 1.0:
            problems.append(f"{label} odds must be greater than 1.00 (got {o}).")
    return problems


def validate_teams(team_a: str, team_b: str) -> list[str]:
    problems = []
    if not team_a or not team_a.strip():
        problems.append("Team A name is required.")
    if not team_b or not team_b.strip():
        problems.append("Team B name is required.")
    if team_a and team_b and team_a.strip().lower() == team_b.strip().lower():
        problems.append("Team A and Team B must be different.")
    return problems
