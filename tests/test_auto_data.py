"""Tests for automatic data: fixture normalization, odds parsing, Elo fallback.

All tests run offline using embedded sample payloads.
"""

import pytest

from api.football_api import is_placeholder, normalize_feed, upcoming_fixtures
from api.odds_api import canonical_name, match_odds_to_fixture, parse_event
from model.batch import recommend_for_match
from model.elo_model import lambdas_from_elo, win_expectancy

SAMPLE_FEED = [
    {"MatchNumber": 1, "RoundNumber": 1, "DateUtc": "2026-06-11 19:00:00Z",
     "Location": "Mexico City Stadium", "HomeTeam": "Mexico", "AwayTeam": "South Africa",
     "Group": "Group A", "HomeTeamScore": None, "AwayTeamScore": None, "Winner": ""},
    {"MatchNumber": 2, "RoundNumber": 1, "DateUtc": "2026-06-12 02:00:00Z",
     "Location": "Guadalajara Stadium", "HomeTeam": "Korea Republic", "AwayTeam": "Czechia",
     "Group": "Group A", "HomeTeamScore": 2, "AwayTeamScore": 0, "Winner": "Korea Republic"},
    {"MatchNumber": 73, "RoundNumber": 4, "DateUtc": "2026-06-28 19:00:00Z",
     "Location": "Los Angeles Stadium", "HomeTeam": "2A", "AwayTeam": "2B",
     "Group": None, "HomeTeamScore": None, "AwayTeamScore": None, "Winner": ""},
    {"MatchNumber": 103, "RoundNumber": 8, "DateUtc": "2026-07-18 19:00:00Z",
     "Location": "Miami Stadium", "HomeTeam": "To be announced", "AwayTeam": "To be announced",
     "Group": None, "HomeTeamScore": None, "AwayTeamScore": None, "Winner": ""},
    {"MatchNumber": 104, "RoundNumber": 8, "DateUtc": "2026-07-19 19:00:00Z",
     "Location": "New York/New Jersey Stadium", "HomeTeam": "To be announced",
     "AwayTeam": "To be announced", "Group": None,
     "HomeTeamScore": None, "AwayTeamScore": None, "Winner": ""},
]

SAMPLE_ODDS_EVENT = {
    "home_team": "South Korea",
    "away_team": "Czech Republic",
    "commence_time": "2026-06-12T02:00:00Z",
    "bookmakers": [
        {"markets": [
            {"key": "h2h", "outcomes": [
                {"name": "South Korea", "price": 2.50},
                {"name": "Czech Republic", "price": 2.90},
                {"name": "Draw", "price": 3.10},
            ]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "point": 2.5, "price": 2.05},
                {"name": "Under", "point": 2.5, "price": 1.78},
            ]},
        ]},
    ],
}


class TestFixtureNormalization:
    def test_phases_assigned_correctly(self):
        df = normalize_feed(SAMPLE_FEED)
        by_id = df.set_index("match_id")
        assert by_id.loc[1, "phase"] == "Group Stage"
        assert by_id.loc[73, "phase"] == "Round of 32"
        assert by_id.loc[103, "phase"] == "Third-place match"
        assert by_id.loc[104, "phase"] == "Final"

    def test_played_flag_and_scores(self):
        df = normalize_feed(SAMPLE_FEED).set_index("match_id")
        assert not df.loc[1, "played"]
        assert df.loc[2, "played"]
        assert df.loc[2, "score_a"] == 2

    def test_placeholders_detected(self):
        assert is_placeholder("2A")
        assert is_placeholder("3ABCDF")
        assert is_placeholder("To be announced")
        assert not is_placeholder("Mexico")
        assert not is_placeholder("Côte d'Ivoire")

    def test_upcoming_excludes_played_and_placeholders(self):
        df = normalize_feed(SAMPLE_FEED)
        up = upcoming_fixtures(df)
        assert list(up["match_id"]) == [1]


class TestOddsParsing:
    def test_team_aliases(self):
        assert canonical_name("South Korea") == "Korea Republic"
        assert canonical_name("Turkey") == "Türkiye"
        assert canonical_name("Ivory Coast") == "Côte d'Ivoire"
        assert canonical_name("France") == "France"

    def test_parse_event_extracts_median_odds(self):
        parsed = parse_event(SAMPLE_ODDS_EVENT)
        assert parsed["team_a"] == "Korea Republic"
        assert parsed["team_b"] == "Czechia"
        assert parsed["odds_a"] == pytest.approx(2.50)
        assert parsed["odds_over_2_5"] == pytest.approx(2.05)

    def test_match_odds_swaps_orientation(self):
        events = [parse_event(SAMPLE_ODDS_EVENT)]
        # Fixture lists Czechia first -> odds must be swapped
        m = match_odds_to_fixture(events, "Czechia", "Korea Republic")
        assert m["odds_a"] == pytest.approx(2.90)
        assert m["odds_b"] == pytest.approx(2.50)
        assert match_odds_to_fixture(events, "France", "Brazil") is None

    def test_incomplete_event_skipped(self):
        event = {**SAMPLE_ODDS_EVENT, "bookmakers": []}
        assert parse_event(event) is None


class TestEloFallback:
    def test_win_expectancy_basics(self):
        assert win_expectancy(2000, 2000) == pytest.approx(0.5)
        assert win_expectancy(2100, 1900) > 0.7

    def test_stronger_team_gets_higher_lambda(self):
        lam_a, lam_b = lambdas_from_elo(2150, 1500)
        assert lam_a > 2.0
        assert lam_b < 0.6
        # symmetric
        lam_b2, lam_a2 = lambdas_from_elo(1500, 2150)
        assert lam_a2 == pytest.approx(lam_a)

    def test_equal_teams_split_evenly(self):
        lam_a, lam_b = lambdas_from_elo(1800, 1800)
        assert lam_a == pytest.approx(lam_b)
        assert lam_a + lam_b == pytest.approx(2.5)

    def test_lambdas_always_positive(self):
        lam_a, lam_b = lambdas_from_elo(2400, 1200)
        assert lam_b > 0


class TestBatchPipeline:
    def test_recommend_with_elo_fallback(self):
        row = recommend_for_match(
            "Argentina", "Haiti", "Group Stage",
            odds=None, elo_ratings={"Argentina": 2150, "Haiti": 1500},
            n_simulations=10_000, seed=1,
        )
        assert row["source"] == "Elo fallback"
        assert row["recommended_pred_a"] > row["recommended_pred_b"]
        assert row["expected_points"] > 0

    def test_recommend_with_market_odds(self):
        odds = {"odds_a": 2.10, "odds_draw": 3.30, "odds_b": 3.60,
                "odds_over_2_5": 1.90, "odds_under_2_5": 1.95}
        row = recommend_for_match(
            "France", "Senegal", "Group Stage",
            odds=odds, n_simulations=10_000, seed=1,
        )
        assert row["source"] == "market odds"
        assert row["odds_a"] == 2.10
