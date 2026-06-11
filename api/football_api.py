"""Real fixture fetching for the FIFA World Cup 2026.

Uses the free fixture feed from fixturedownload.com (no API key required).
The feed contains all 104 matches with the real draw, kickoff times (UTC),
venues, and live scores once matches are played. Knockout slots that are not
yet decided appear as placeholders like "1A", "2B", "3ABCDF".
"""

from pathlib import Path

import pandas as pd
import requests

FEED_URL = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"
FIXTURES_FILE = Path(__file__).resolve().parent.parent / "data" / "fixtures.csv"

ROUND_TO_PHASE = {
    1: "Group Stage",
    2: "Group Stage",
    3: "Group Stage",
    4: "Round of 32",
    5: "Round of 16",
    6: "Quarterfinal",
    7: "Semifinal",
    # Round 8 holds two matches: the earlier one is the third-place match,
    # the later one is the final (resolved in _phase_for_round_8).
}

PLACEHOLDER_MARKERS = ("To be announced",)


def is_placeholder(team: str) -> bool:
    """True for undecided knockout slots like '1A', '2B', '3ABCDF'."""
    if not team or team in PLACEHOLDER_MARKERS:
        return True
    return team[0].isdigit()


def fetch_fixtures(timeout: int = 20) -> pd.DataFrame:
    """Download all WC2026 fixtures and normalize them to the app schema.

    Returns a DataFrame with columns:
        match_id, date, kickoff_utc, phase, group, team_a, team_b,
        venue, score_a, score_b, played
    """
    response = requests.get(FEED_URL, timeout=timeout)
    response.raise_for_status()
    return normalize_feed(response.json())


def normalize_feed(feed: list[dict]) -> pd.DataFrame:
    rows = []
    round_8 = sorted(
        (m for m in feed if m.get("RoundNumber") == 8),
        key=lambda m: m["DateUtc"],
    )
    third_place_id = round_8[0]["MatchNumber"] if len(round_8) == 2 else None

    for m in feed:
        rnd = m.get("RoundNumber")
        if rnd == 8:
            phase = "Third-place match" if m["MatchNumber"] == third_place_id else "Final"
        else:
            phase = ROUND_TO_PHASE.get(rnd, "Group Stage")

        dt = pd.to_datetime(m["DateUtc"], utc=True)
        played = m.get("HomeTeamScore") is not None and m.get("AwayTeamScore") is not None
        rows.append(
            {
                "match_id": m["MatchNumber"],
                "date": dt.strftime("%Y-%m-%d"),
                "kickoff_utc": dt.strftime("%H:%M"),
                "phase": phase,
                "group": (m.get("Group") or "").replace("Group ", ""),
                "team_a": m["HomeTeam"],
                "team_b": m["AwayTeam"],
                "venue": m.get("Location", ""),
                "score_a": m.get("HomeTeamScore"),
                "score_b": m.get("AwayTeamScore"),
                "played": played,
            }
        )
    return pd.DataFrame(rows).sort_values("match_id", ignore_index=True)


def refresh_fixtures_file() -> pd.DataFrame:
    """Download fixtures and persist them to data/fixtures.csv."""
    df = fetch_fixtures()
    FIXTURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIXTURES_FILE, index=False)
    return df


def load_or_fetch_fixtures() -> pd.DataFrame:
    """Load fixtures from the local CSV; download them on first use."""
    if FIXTURES_FILE.exists():
        df = pd.read_csv(FIXTURES_FILE)
        if "kickoff_utc" in df.columns:  # new schema
            return df
    return refresh_fixtures_file()


def upcoming_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Matches not yet played where both teams are decided."""
    mask = (
        ~fixtures["played"].astype(bool)
        & ~fixtures["team_a"].map(is_placeholder)
        & ~fixtures["team_b"].map(is_placeholder)
    )
    return fixtures[mask].copy()


# --- Not yet implemented (v1.3 roadmap) ---------------------------------


def fetch_team_info(team: str) -> dict:
    raise NotImplementedError("Use data/teams.csv in the MVP.")


def fetch_lineups(match: dict) -> dict:
    raise NotImplementedError


def fetch_injuries(team: str) -> list[dict]:
    raise NotImplementedError


def fetch_suspensions(team: str) -> list[dict]:
    raise NotImplementedError


def fetch_recent_form(team: str, n_matches: int = 5) -> list[dict]:
    raise NotImplementedError
