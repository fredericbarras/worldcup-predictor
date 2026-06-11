"""Placeholder for team news fetching and summarization.

Future idea: fetch recent news headlines for a team and summarize anything
relevant to the match (injuries, rotation, morale) to help fill the manual
adjustment fields with real information.
"""


def fetch_team_news(team: str, max_items: int = 10) -> list[dict]:
    """Return [{"title", "url", "published_at", "source"}]."""
    raise NotImplementedError("News fetching is not implemented in the MVP.")


def summarize_relevant_news(team: str) -> str:
    """Return a short text summary of match-relevant news for a team."""
    raise NotImplementedError("News summarization is not implemented in the MVP.")
