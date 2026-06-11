# ⚽ World Cup 2026 Prediction Optimizer

A Monte Carlo match simulator that recommends the scoreline prediction that
**maximizes expected points under your family competition rules** — not just
the most likely exact score.

## Why this exists

In most prediction competitions, points come from getting the **winner**, the
**goal difference**, and the **total goals** right — the exact score is just
the jackpot. The mathematically best prediction is therefore often *not* the
most probable score. This tool starts from betting odds (the best freely
available forecast), turns them into an expected-goals model, simulates the
match tens of thousands of times, scores every candidate prediction under the
family rules, and recommends the expected-points maximizer.

## Quick start

```bash
cd worldcup_predictor
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. No API key is required: fixtures are
downloaded automatically and an Elo fallback model covers matches without
odds. A free Odds API key makes the predictions sharper.

## Zero-input workflow (all matches at once)

1. Open the **All Matches** tab.
2. (Optional, recommended) In the sidebar, paste a free key from
   [the-odds-api.com](https://the-odds-api.com) and click **Fetch odds for
   all matches** — one request covers every upcoming match.
3. Click **Simulate all upcoming matches**.
4. Read the **✅ Predict** column: one recommended score per match.
5. Download the CSV or save everything to the prediction log.

Without an odds key, the Elo fallback model is used automatically — no
manual input of any kind is needed.

## Single-match deep dive (before kickoff)

1. Pick the match in **Load a fixture** — teams, phase, date, and (if
   fetched) odds are filled in automatically.
2. Optionally add small xG adjustments for confirmed lineups, injuries, etc.
3. Click **Run simulation**, submit the **recommended prediction**, and
   click **Save this prediction** to log it.

## Data sources

- **Fixtures**: full real 104-match WC2026 schedule from
  fixturedownload.com (free, no key), including live scores; refresh from
  the All Matches tab. Knockout slots appear once the bracket is decided.
- **Odds**: The Odds API (free tier: 500 requests/month; one request
  returns all matches). Median odds across bookmakers.
- **Elo fallback**: `data/elo_ratings.csv` — bundled snapshot of all 48
  qualified teams; update from [eloratings.net](https://www.eloratings.net)
  for best results.

## Scoring rules implemented

| | Winner/draw | Total goals | Goal difference* |
|---|---|---|---|
| Group stage | 5 | 1 | 3 |
| Knockout (R32 → final, after 120 min) | 10 | 2 | 6 |

\* Goal-difference points require the winner to be correct; per the rules
("le gagnant doit être correct") the bonus is **not** awarded for draws.
This assumption is a single flag in `model/scoring.py`
(`DRAW_EARNS_DIFF_BONUS = False`).

Knockout predictions target the result **after 120 minutes** — penalty
shootout goals are excluded, so a draw is a valid knockout prediction.

## How the model works

0. **Model source per match** — fetched/manual odds when available (best),
   otherwise the Elo fallback: win expectancy `1/(1+10^(−Δelo/400))`,
   expected goal margin `Δelo/220` (capped ±3), total goals
   `2.5 + 0.25·|margin|` (see `model/elo_model.py`).
1. **Margin removal** — implied probabilities (`1/odds`) are normalized so
   they sum to 1, removing the bookmaker overround.
2. **Total expected goals** — root-find the Poisson mean whose
   P(goals > 2.5) matches the over/under market.
3. **Split between teams** — SciPy optimization finds `(λ_A, λ_B)` whose
   Poisson model best reproduces the market's win/draw/loss/over-2.5
   probabilities (bounds 0.1–5.0 per team).
4. **Transparent adjustments** — optional bounded xG nudges (form, injuries,
   lineups, rest, motivation, host advantage, Elo/FIFA gaps), documented in
   `model/adjustments.py`.
5. **Monte Carlo** — N independent Poisson scorelines (default 50,000).
6. **Expected-points optimization** — every candidate prediction (0-0 … 6-6
   by default, extensible to 10-10) is scored against every simulation under
   the family rules; the highest average wins the recommendation.

Full explanation in the app's **Methodology** tab.

## Project structure

```
worldcup_predictor/
├── app.py                  # Streamlit app
├── requirements.txt
├── data/                   # local CSV storage (teams, fixtures, Elo, predictions)
├── model/
│   ├── odds.py             # odds → fair probabilities
│   ├── expected_goals.py   # market → (λ_A, λ_B) via root-finding + optimization
│   ├── elo_model.py        # Elo fallback (zero-input expected goals)
│   ├── poisson.py          # Poisson score-grid helpers
│   ├── adjustments.py      # bounded, documented xG adjustments
│   ├── monte_carlo.py      # simulation engine
│   ├── scoring.py          # family competition rules
│   ├── optimizer.py        # expected-points maximization
│   └── batch.py            # all-matches batch recommendations
├── api/
│   ├── football_api.py     # real fixture download (fixturedownload.com)
│   ├── odds_api.py         # real odds fetching (The Odds API)
│   └── news_api.py         # placeholder (v1.3)
└── tests/                  # pytest suite
```

## Running the tests

```bash
python -m pytest tests/ -v
```

## Data files

- `data/teams.csv` — sample team list with FIFA ranking / Elo placeholders.
  Update with real values as the tournament approaches.
- `data/fixtures.csv` — sample fixtures (placeholder schedule).
- `data/elo_ratings.csv` — Elo snapshot (update from eloratings.net).
- `data/predictions.csv` — your saved predictions (appended by the app).
- `data/sample_inputs.csv` — example odds scenarios to try the app with.

## Roadmap

- **v1.1** ✅ shipped — automatic odds fetching (The Odds API), real
  fixtures, Elo fallback model, one-click fixture loading, batch
  simulation of all upcoming matches.
- **v1.2** — full tournament simulation: group tables, qualification and
  champion probabilities, subsidiary-question optimizer (50 pts champion!).
- **v1.3** — lineup/injury feeds, news summarization, a "late update" mode
  for the final 30 minutes before kickoff.
- **v1.4** — strategic modes vs. other family members: conservative,
  contrarian, and catch-up strategies based on the leaderboard.

The `api/` modules already define the function signatures for v1.1–v1.3;
they raise `NotImplementedError` in the MVP.
