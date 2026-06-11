"""Automatic odds fetching via The Odds API (https://the-odds-api.com).

A free account (500 requests/month) is enough for the whole tournament:
one request returns 1X2 + totals odds for every upcoming World Cup match.

Set the key in the app sidebar or via the ODDS_API_KEY environment variable.
"""

import os
import statistics

import requests

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"

# The Odds API uses slightly different team names than the FIFA fixture feed.
# Keys: Odds API name -> fixture feed name.
TEAM_ALIASES = {
    "South Korea": "Korea Republic",
    "Korea": "Korea Republic",
    "Iran": "IR Iran",
    "Turkey": "Türkiye",
    "United States": "USA",
    "United States of America": "USA",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Cape Verde Islands": "Cabo Verde",
    "Czech Republic": "Czechia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
}


def canonical_name(team: str) -> str:
    return TEAM_ALIASES.get(team.strip(), team.strip())


def get_api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY") or None


def fetch_all_match_odds(api_key: str, timeout: int = 20) -> list[dict]:
    """Fetch 1X2 and Over/Under 2.5 odds for all upcoming WC matches.

    Returns one dict per match:
        {"team_a", "team_b", "commence_time",
         "odds_a", "odds_draw", "odds_b", "odds_over_2_5", "odds_under_2_5",
         "n_bookmakers"}

    Odds are the median across bookmakers (robust to outliers). Matches
    missing either market are skipped.
    """
    response = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/odds",
        params={
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return [m for m in (parse_event(e) for e in response.json()) if m]


def parse_event(event: dict) -> dict | None:
    home = canonical_name(event["home_team"])
    away = canonical_name(event["away_team"])

    h2h = {"home": [], "draw": [], "away": []}
    totals = {"over": [], "under": []}

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    name = canonical_name(outcome["name"])
                    if name == home:
                        h2h["home"].append(outcome["price"])
                    elif name == away:
                        h2h["away"].append(outcome["price"])
                    elif outcome["name"] == "Draw":
                        h2h["draw"].append(outcome["price"])
            elif market["key"] == "totals":
                for outcome in market["outcomes"]:
                    if outcome.get("point") == 2.5:
                        side = outcome["name"].lower()
                        if side in totals:
                            totals[side].append(outcome["price"])

    if not all(h2h.values()) or not all(totals.values()):
        return None

    return {
        "team_a": home,
        "team_b": away,
        "commence_time": event.get("commence_time", ""),
        "odds_a": statistics.median(h2h["home"]),
        "odds_draw": statistics.median(h2h["draw"]),
        "odds_b": statistics.median(h2h["away"]),
        "odds_over_2_5": statistics.median(totals["over"]),
        "odds_under_2_5": statistics.median(totals["under"]),
        "n_bookmakers": len(event.get("bookmakers", [])),
    }


def match_odds_to_fixture(odds_events: list[dict], team_a: str, team_b: str) -> dict | None:
    """Find the odds event for a fixture, regardless of home/away orientation.

    If the bookmaker lists the teams in the opposite order, the odds are
    swapped so that odds_a always refers to the fixture's team_a.
    """
    a, b = canonical_name(team_a), canonical_name(team_b)
    for event in odds_events:
        if event["team_a"] == a and event["team_b"] == b:
            return event
        if event["team_a"] == b and event["team_b"] == a:
            return {
                **event,
                "team_a": a,
                "team_b": b,
                "odds_a": event["odds_b"],
                "odds_b": event["odds_a"],
            }
    return None


# Backwards-compatible single-match helpers (v1 API surface).


def fetch_1x2_odds(match: dict) -> dict:
    raise NotImplementedError("Use fetch_all_match_odds(api_key) instead.")


def fetch_over_under_odds(match: dict, line: float = 2.5) -> dict:
    raise NotImplementedError("Use fetch_all_match_odds(api_key) instead.")
