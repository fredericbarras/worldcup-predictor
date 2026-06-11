"""World Cup 2026 Prediction Optimizer — Streamlit app.

Run with:
    streamlit run app.py
"""

import hmac
import os
from datetime import datetime, date, time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api.football_api import (
    load_or_fetch_fixtures,
    refresh_fixtures_file,
    upcoming_fixtures,
)
from api.odds_api import fetch_all_match_odds, get_api_key, match_odds_to_fixture
from model.adjustments import AdjustmentInputs, apply_adjustments, compute_adjustments
from model.batch import simulate_all_upcoming
from model.elo_model import lambdas_for_teams, load_elo_ratings
from model.expected_goals import fit_lambdas
from model.monte_carlo import simulate_match
from model.odds import market_probabilities
from model.optimizer import best_prediction, evaluate_all_predictions, most_likely_score
from model.scoring import ALL_PHASES, get_weights
from utils import storage
from utils.formatting import pct, score_label, signed
from utils.validation import validate_odds, validate_teams

st.set_page_config(
    page_title="World Cup 2026 Prediction Optimizer",
    page_icon="⚽",
    layout="wide",
)


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets (cloud) or environment variables (local)."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass  # no secrets.toml configured — fall back to the environment
    return os.environ.get(name, default)


def require_password() -> None:
    """Block the app behind a password when APP_PASSWORD is configured.

    If no APP_PASSWORD secret/env var is set (e.g. running locally),
    the app is open — no password prompt.
    """
    expected = get_secret("APP_PASSWORD")
    if not expected or st.session_state.get("authenticated"):
        return
    st.title("⚽ World Cup 2026 Prediction Optimizer")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


require_password()

st.title("⚽ World Cup 2026 Prediction Optimizer")
st.caption("Monte Carlo simulator optimized for family competition scoring rules.")


# ------------------------------------------------------------ shared data ---

try:
    fixtures_df = load_or_fetch_fixtures()
except Exception as exc:  # offline and no cached file
    st.warning(f"Could not load fixtures: {exc}")
    fixtures_df = pd.DataFrame(
        columns=["match_id", "date", "kickoff_utc", "phase", "group",
                 "team_a", "team_b", "venue", "score_a", "score_b", "played"]
    )

upcoming_df = upcoming_fixtures(fixtures_df) if not fixtures_df.empty else fixtures_df

teams_df = storage.load_teams()
team_names = sorted(teams_df["team_name"].dropna().tolist()) if not teams_df.empty else []

MANUAL_ENTRY = "Manual entry"


def fixture_label(fx) -> str:
    grp = f", Group {fx['group']}" if fx["group"] else ""
    return (
        f"{fx['date']} {fx['kickoff_utc']} UTC · "
        f"{fx['team_a']} vs {fx['team_b']} ({fx['phase']}{grp})"
    )


fixture_map = {fixture_label(fx): fx for _, fx in upcoming_df.iterrows()}
st.session_state["fixture_map"] = fixture_map


def _on_fixture_selected():
    """Prefill match setup (and odds, if fetched) from the chosen fixture."""
    fx = st.session_state["fixture_map"].get(st.session_state.get("fixture_sel"))
    if fx is None:
        return
    st.session_state["team_a_sel"] = fx["team_a"]
    st.session_state["team_b_sel"] = fx["team_b"]
    st.session_state["phase_sel"] = fx["phase"]
    st.session_state["match_date_in"] = date.fromisoformat(fx["date"])
    st.session_state["kickoff_in"] = time.fromisoformat(fx["kickoff_utc"] + ":00")
    event = match_odds_to_fixture(
        st.session_state.get("odds_events") or [], fx["team_a"], fx["team_b"]
    )
    if event:
        st.session_state["odds_a_in"] = float(event["odds_a"])
        st.session_state["odds_draw_in"] = float(event["odds_draw"])
        st.session_state["odds_b_in"] = float(event["odds_b"])
        st.session_state["odds_over_in"] = float(event["odds_over_2_5"])
        st.session_state["odds_under_in"] = float(event["odds_under_2_5"])
        st.session_state["odds_prefilled_from_api"] = True
    else:
        st.session_state["odds_prefilled_from_api"] = False


# ---------------------------------------------------------------- sidebar ---

with st.sidebar:
    st.header("0 · Automatic data")
    st.selectbox(
        "Load a fixture (auto-fills everything below)",
        [MANUAL_ENTRY] + list(fixture_map.keys()),
        key="fixture_sel",
        on_change=_on_fixture_selected,
    )
    with st.expander("Odds auto-fetch (The Odds API)"):
        st.caption(
            "Free key at the-odds-api.com (500 requests/month — one fetch "
            "covers all matches). Or set the ODDS_API_KEY environment variable."
        )
        api_key = st.text_input(
            "API key",
            value=get_secret("ODDS_API_KEY") or get_api_key() or "",
            type="password",
        )
        if st.button("📡 Fetch odds for all matches"):
            if not api_key:
                st.error("Enter an API key first.")
            else:
                try:
                    events = fetch_all_match_odds(api_key)
                    st.session_state["odds_events"] = events
                    st.session_state["odds_fetched_at"] = datetime.now().strftime("%H:%M")
                    st.success(f"Odds fetched for {len(events)} matches.")
                except Exception as exc:
                    st.error(f"Fetch failed: {exc}")
        if st.session_state.get("odds_events"):
            st.caption(
                f"✅ {len(st.session_state['odds_events'])} matches with odds "
                f"(fetched {st.session_state.get('odds_fetched_at', '')})"
            )

    st.header("1 · Match setup")

    def fresh(key: str, **kwargs):
        """Widget default kwargs, only when the key has no session state yet
        (avoids Streamlit's default-vs-session-state warning)."""
        return {} if key in st.session_state else kwargs

    def team_picker(label: str, key: str, default_index: int) -> str:
        options = team_names + ["Other (type below)"]
        choice = st.selectbox(
            label, options, key=key,
            **fresh(key, index=min(default_index, len(options) - 1)),
        )
        if choice == "Other (type below)":
            return st.text_input(f"{label} (custom name)", key=f"custom_{key}")
        return choice

    team_a = team_picker("Team A", "team_a_sel", 0)
    team_b = team_picker("Team B", "team_b_sel", 1)
    phase = st.selectbox("Tournament phase", ALL_PHASES, key="phase_sel")
    match_date = st.date_input(
        "Match date", key="match_date_in", **fresh("match_date_in", value=date(2026, 6, 11))
    )
    kickoff = st.time_input(
        "Kickoff time (UTC)", key="kickoff_in", **fresh("kickoff_in", value=time(18, 0))
    )
    host_team = st.radio(
        "Host advantage",
        ["No host advantage", "Team A hosts", "Team B hosts"],
        help="Applied as an xG bonus (set the magnitude under Adjustments).",
    )

    st.header("2 · Model input")
    odds_source = st.radio(
        "Source",
        ["Betting odds (manual or fetched)", "Elo model (no odds needed)"],
        help="Odds are sharper. The Elo fallback needs no input at all.",
    )
    use_elo = odds_source.startswith("Elo")

    if not use_elo:
        if st.session_state.get("odds_prefilled_from_api"):
            st.caption("✅ Odds below pre-filled from The Odds API (median of bookmakers).")
        col1, col2 = st.columns(2)
        odds_a = col1.number_input("Team A win", min_value=1.01, step=0.05, key="odds_a_in", **fresh("odds_a_in", value=2.10))
        odds_b = col2.number_input("Team B win", min_value=1.01, step=0.05, key="odds_b_in", **fresh("odds_b_in", value=3.60))
        odds_draw = st.number_input("Draw", min_value=1.01, step=0.05, key="odds_draw_in", **fresh("odds_draw_in", value=3.30))
        col3, col4 = st.columns(2)
        odds_over = col3.number_input("Over 2.5", min_value=1.01, step=0.05, key="odds_over_in", **fresh("odds_over_in", value=1.90))
        odds_under = col4.number_input("Under 2.5", min_value=1.01, step=0.05, key="odds_under_in", **fresh("odds_under_in", value=1.95))
    else:
        odds_a = odds_draw = odds_b = odds_over = odds_under = None
        ratings = load_elo_ratings()
        st.caption(
            f"Elo: {team_a} {ratings.get(team_a, 1700):.0f} — "
            f"{ratings.get(team_b, 1700):.0f} {team_b} "
            "(edit data/elo_ratings.csv to update)"
        )

    st.header("3 · Adjustments (optional)")
    with st.expander("Objective xG adjustments", expanded=False):
        st.caption(
            "Small additive xG nudges on top of the baseline. "
            "Leave at 0 to trust the model fully. See the Methodology tab."
        )
        elo_diff = st.number_input(
            "Elo difference (A − B)", value=0.0, step=10.0,
            help="Converted to xG at 0.0008 per Elo point, capped at ±0.25 total.",
        )
        fifa_rank_diff = st.number_input(
            "FIFA ranking difference (B − A)", value=0.0, step=1.0,
            help="Positive = team A better ranked. 0.002 xG per place, cap ±0.10.",
        )
        ca, cb = st.columns(2)
        form_a = ca.slider("Recent form A", -0.15, 0.15, 0.0, 0.01)
        form_b = cb.slider("Recent form B", -0.15, 0.15, 0.0, 0.01)
        injury_a = ca.slider("Injuries/suspensions A", -0.30, 0.30, 0.0, 0.01)
        injury_b = cb.slider("Injuries/suspensions B", -0.30, 0.30, 0.0, 0.01)
        lineup_a = ca.slider("Lineup strength A", -0.30, 0.30, 0.0, 0.01)
        lineup_b = cb.slider("Lineup strength B", -0.30, 0.30, 0.0, 0.01)
        rest_a = ca.slider("Rest/fatigue A", -0.10, 0.10, 0.0, 0.01)
        rest_b = cb.slider("Rest/fatigue B", -0.10, 0.10, 0.0, 0.01)
        motivation_a = ca.slider("Motivation A", -0.15, 0.15, 0.0, 0.01)
        motivation_b = cb.slider("Motivation B", -0.15, 0.15, 0.0, 0.01)
        host_advantage = st.slider(
            "Host advantage magnitude (xG)", 0.0, 0.25, 0.15, 0.01,
            help="Only applied if a host team is selected above.",
        )

    st.header("4 · Simulation settings")
    n_simulations = st.selectbox(
        "Monte Carlo simulations", [10_000, 50_000, 100_000, 250_000], index=1
    )
    max_pred_goals = st.slider("Max goals per team (candidates/display)", 4, 10, 6)
    seed_input = st.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
    seed = int(seed_input) if seed_input > 0 else None

    notes = st.text_area("Notes (saved with the prediction)", "")

    run = st.button("🎲 Run simulation", type="primary", use_container_width=True)


# ------------------------------------------------------------ run pipeline ---

if run:
    problems = validate_teams(team_a, team_b)
    if not use_elo:
        problems += validate_odds(odds_a, odds_draw, odds_b, odds_over, odds_under)
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    adjustments = AdjustmentInputs(
        elo_diff=elo_diff,
        fifa_rank_diff=fifa_rank_diff,
        recent_form_a=form_a,
        recent_form_b=form_b,
        injury_a=injury_a,
        injury_b=injury_b,
        lineup_a=lineup_a,
        lineup_b=lineup_b,
        rest_a=rest_a,
        rest_b=rest_b,
        motivation_a=motivation_a,
        motivation_b=motivation_b,
        host_advantage=host_advantage,
        host_team={"Team A hosts": "a", "Team B hosts": "b"}.get(host_team, "none"),
    )

    with st.spinner("Fitting model and simulating..."):
        if use_elo:
            lam_a_base, lam_b_base, _, _ = lambdas_for_teams(team_a, team_b)
            market, estimate = None, None
        else:
            market = market_probabilities(odds_a, odds_draw, odds_b, odds_over, odds_under)
            estimate = fit_lambdas(market)
            lam_a_base, lam_b_base = estimate.lambda_a, estimate.lambda_b
        lam_a_adj, lam_b_adj = apply_adjustments(lam_a_base, lam_b_base, adjustments)
        sim = simulate_match(lam_a_adj, lam_b_adj, n_simulations, seed)
        evaluation = evaluate_all_predictions(sim, phase, max_pred_goals)

    st.session_state["results"] = {
        "team_a": team_a,
        "team_b": team_b,
        "phase": phase,
        "kickoff_time": f"{match_date} {kickoff}",
        "odds": (odds_a, odds_draw, odds_b, odds_over, odds_under),
        "market": market,
        "estimate": estimate,
        "lam_a_base": lam_a_base,
        "lam_b_base": lam_b_base,
        "adjustments": adjustments,
        "lam_a_adj": lam_a_adj,
        "lam_b_adj": lam_b_adj,
        "sim": sim,
        "evaluation": evaluation,
        "max_pred_goals": max_pred_goals,
        "n_simulations": n_simulations,
        "source": "Elo fallback" if use_elo else "market odds",
        "notes": notes,
    }


# ----------------------------------------------------------------- display ---

tabs = st.tabs(
    [
        "📅 All Matches",
        "🏆 Recommendation",
        "📊 Probabilities",
        "🔢 Score Matrix",
        "🎯 Expected Points",
        "💾 Saved Predictions",
        "📖 Methodology",
    ]
)

results = st.session_state.get("results")

# --------------------------------------------------------- All Matches tab ---
with tabs[0]:
    top_l, top_r = st.columns([3, 1])
    top_l.subheader("Tournament schedule & batch recommendations")
    if top_r.button("🔄 Refresh fixtures from web"):
        try:
            fixtures_df = refresh_fixtures_file()
            upcoming_df = upcoming_fixtures(fixtures_df)
            st.success(f"Fixtures refreshed — {len(fixtures_df)} matches.")
            st.rerun()
        except Exception as exc:
            st.error(f"Refresh failed: {exc}")

    n_upcoming = len(upcoming_df)
    n_played = int(fixtures_df["played"].astype(bool).sum()) if not fixtures_df.empty else 0
    odds_events = st.session_state.get("odds_events")
    st.caption(
        f"{len(fixtures_df)} fixtures · {n_played} played · {n_upcoming} upcoming with "
        f"decided teams · odds available for "
        f"{len(odds_events) if odds_events else 0} matches "
        f"({'fetch them in the sidebar to sharpen the model' if not odds_events else 'fetched'})"
    )

    bc1, bc2, bc3 = st.columns([1, 1, 2])
    batch_sims = bc1.selectbox("Simulations per match", [10_000, 20_000, 50_000], index=1)
    horizon = bc2.selectbox("Matches to simulate", ["Next 7 days", "Next 3 days", "All upcoming"], index=0)
    run_batch = bc3.button("🎲 Simulate all upcoming matches", type="primary")

    if run_batch:
        batch_fixtures = upcoming_df
        if horizon != "All upcoming" and not upcoming_df.empty:
            days = 7 if "7" in horizon else 3
            cutoff = pd.Timestamp.now().normalize() + pd.Timedelta(days=days)
            batch_fixtures = upcoming_df[pd.to_datetime(upcoming_df["date"]) <= cutoff]
        if batch_fixtures.empty:
            st.warning("No upcoming matches in the selected window.")
        else:
            bar = st.progress(0.0, text="Simulating...")
            batch_results = simulate_all_upcoming(
                batch_fixtures,
                odds_events=odds_events,
                n_simulations=batch_sims,
                progress_callback=lambda f, label: bar.progress(f, text=f"Simulating {label}"),
            )
            bar.empty()
            st.session_state["batch_results"] = batch_results

    batch_results = st.session_state.get("batch_results")
    if batch_results is not None:
        st.markdown("#### Recommended predictions")
        show = batch_results.copy()
        show["match"] = show["team_a"] + " — " + show["team_b"]
        show["win/draw/loss"] = (
            (show["p_a_win"] * 100).round(0).astype(int).astype(str) + "/"
            + (show["p_draw"] * 100).round(0).astype(int).astype(str) + "/"
            + (show["p_b_win"] * 100).round(0).astype(int).astype(str) + "%"
        )
        st.dataframe(
            show[
                ["date", "kickoff_utc", "match", "phase", "source",
                 "recommended", "expected_points", "most_likely", "win/draw/loss"]
            ].rename(columns={
                "date": "Date", "kickoff_utc": "Kickoff (UTC)", "match": "Match",
                "phase": "Phase", "source": "Model source", "recommended": "✅ Predict",
                "expected_points": "Expected pts", "most_likely": "Most likely",
            }),
            use_container_width=True,
            hide_index=True,
            height=min(38 * (len(show) + 1), 700),
        )
        dl, sv = st.columns(2)
        dl.download_button(
            "⬇️ Download recommendations CSV",
            batch_results.to_csv(index=False).encode("utf-8"),
            file_name="batch_recommendations.csv",
            mime="text/csv",
        )
        if sv.button("💾 Save all to prediction log"):
            for _, r in batch_results.iterrows():
                storage.append_prediction(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "team_a": r["team_a"],
                        "team_b": r["team_b"],
                        "phase": r["phase"],
                        "kickoff_time": f"{r['date']} {r['kickoff_utc']}",
                        "odds_a": r["odds_a"],
                        "odds_draw": r["odds_draw"],
                        "odds_b": r["odds_b"],
                        "odds_over_2_5": r["odds_over_2_5"],
                        "odds_under_2_5": r["odds_under_2_5"],
                        "lambda_a_baseline": r["lambda_a"],
                        "lambda_b_baseline": r["lambda_b"],
                        "lambda_a_adjusted": r["lambda_a"],
                        "lambda_b_adjusted": r["lambda_b"],
                        "simulations": batch_sims,
                        "recommended_pred_a": r["recommended_pred_a"],
                        "recommended_pred_b": r["recommended_pred_b"],
                        "expected_points": r["expected_points"],
                        "most_likely_score_a": int(r["most_likely"].split("-")[0]),
                        "most_likely_score_b": int(r["most_likely"].split("-")[1]),
                        "most_likely_score_probability": r["most_likely_prob"],
                        "sim_p_a_win": r["p_a_win"],
                        "sim_p_draw": r["p_draw"],
                        "sim_p_b_win": r["p_b_win"],
                        "notes": f"batch ({r['source']})",
                    }
                )
            st.success(f"Saved {len(batch_results)} predictions to data/predictions.csv ✅")
    else:
        st.info(
            "Click **Simulate all upcoming matches** to get a recommendation for "
            "every match at once — with zero manual input. Without an odds API key "
            "the Elo fallback model is used; fetch odds in the sidebar for sharper "
            "predictions. For a single match deep-dive (charts, score matrix, "
            "adjustments), load a fixture in the sidebar and run a simulation."
        )

    with st.expander("Full schedule"):
        st.dataframe(fixtures_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------- Recommendation tab ---
with tabs[1]:
    if results is None:
        st.info(
            "Load a fixture in the sidebar (or set up a match manually) and click "
            "**Run simulation** for a single-match deep dive."
        )
    else:
        sim = results["sim"]
        evaluation = results["evaluation"]
        best = best_prediction(evaluation)
        likely = most_likely_score(evaluation)
        team_a, team_b = results["team_a"], results["team_b"]
        weights = get_weights(results["phase"])

        st.subheader(f"{team_a} vs {team_b} — {results['phase']}")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "✅ Recommended prediction",
            score_label(best["pred_a"], best["pred_b"]),
            f"{best['expected_points']:.2f} expected points",
        )
        c2.metric(
            "Most likely exact score",
            score_label(likely["pred_a"], likely["pred_b"]),
            f"{pct(likely['p_exact'])} probability",
        )
        c3.metric(
            "Max points possible",
            f"{weights.max_points}",
            f"{weights.winner_points}/{weights.total_goals_points}/{weights.goal_diff_points} winner/total/diff",
        )

        c4, c5, c6, c7, c8 = st.columns(5)
        c4.metric(f"P({team_a} win)", pct(sim.p_a_win))
        c5.metric("P(draw)", pct(sim.p_draw))
        c6.metric(f"P({team_b} win)", pct(sim.p_b_win))
        c7.metric(f"xG {team_a}", f"{results['lam_a_adj']:.2f}")
        c8.metric(f"xG {team_b}", f"{results['lam_b_adj']:.2f}")

        same = (best["pred_a"], best["pred_b"]) == (likely["pred_a"], likely["pred_b"])
        if same:
            st.success(
                "Here the most likely exact score is **also** the best prediction "
                "under your family rules — no trade-off this time."
            )
        else:
            st.info(
                f"**Why {score_label(best['pred_a'], best['pred_b'])} and not "
                f"{score_label(likely['pred_a'], likely['pred_b'])}?** "
                f"{score_label(likely['pred_a'], likely['pred_b'])} is the most likely "
                f"single score ({pct(likely['p_exact'])}), but "
                f"{score_label(best['pred_a'], best['pred_b'])} earns more points on "
                f"average ({best['expected_points']:.2f} vs "
                f"{likely['expected_points']:.2f}) because it captures the correct "
                f"winner in {pct(best['p_winner_correct'])} of simulations and the "
                f"correct goal difference in {pct(best['p_diff_correct'])} — and under "
                f"your rules those bonuses outweigh the exact-score probability gap."
            )

        est = results["estimate"]
        adj_a, adj_b = compute_adjustments(results["adjustments"])
        with st.expander("Model internals"):
            st.write(
                pd.DataFrame(
                    {
                        "metric": [
                            f"Baseline λ ({results['source']})",
                            "Manual adjustment (xG)",
                            "Adjusted λ (used in simulation)",
                        ],
                        team_a: [f"{results['lam_a_base']:.3f}", signed(adj_a), f"{results['lam_a_adj']:.3f}"],
                        team_b: [f"{results['lam_b_base']:.3f}", signed(adj_b), f"{results['lam_b_adj']:.3f}"],
                    }
                )
            )
            if results["market"] is not None:
                m = results["market"]
                st.caption(
                    f"Market fit error: {est.fit_error:.2e} · "
                    f"1X2 overround removed: {pct(m.overround_1x2)} · "
                    f"O/U overround removed: {pct(m.overround_ou)} · "
                    f"Simulations: {results['n_simulations']:,}"
                )
            else:
                st.caption(
                    f"Elo fallback model (no odds) · Simulations: {results['n_simulations']:,}"
                )

        if st.button("💾 Save this prediction", type="primary"):
            o = results["odds"]
            storage.append_prediction(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "team_a": team_a,
                    "team_b": team_b,
                    "phase": results["phase"],
                    "kickoff_time": results["kickoff_time"],
                    "odds_a": o[0],
                    "odds_draw": o[1],
                    "odds_b": o[2],
                    "odds_over_2_5": o[3],
                    "odds_under_2_5": o[4],
                    "lambda_a_baseline": round(results["lam_a_base"], 4),
                    "lambda_b_baseline": round(results["lam_b_base"], 4),
                    "lambda_a_adjusted": round(results["lam_a_adj"], 4),
                    "lambda_b_adjusted": round(results["lam_b_adj"], 4),
                    "simulations": results["n_simulations"],
                    "recommended_pred_a": int(best["pred_a"]),
                    "recommended_pred_b": int(best["pred_b"]),
                    "expected_points": round(float(best["expected_points"]), 4),
                    "most_likely_score_a": int(likely["pred_a"]),
                    "most_likely_score_b": int(likely["pred_b"]),
                    "most_likely_score_probability": round(float(likely["p_exact"]), 4),
                    "sim_p_a_win": round(sim.p_a_win, 4),
                    "sim_p_draw": round(sim.p_draw, 4),
                    "sim_p_b_win": round(sim.p_b_win, 4),
                    "notes": results["notes"] or results["source"],
                }
            )
            st.success("Prediction saved to data/predictions.csv ✅")

# --------------------------------------------------------- Probabilities tab ---
with tabs[2]:
    if results is None:
        st.info("Run a single-match simulation first.")
    else:
        sim = results["sim"]
        team_a, team_b = results["team_a"], results["team_b"]

        left, right = st.columns(2)
        with left:
            st.subheader("Match outcome")
            outcome_df = pd.DataFrame(
                {
                    "Outcome": [f"{team_a} win", "Draw", f"{team_b} win"],
                    "Probability": [sim.p_a_win, sim.p_draw, sim.p_b_win],
                }
            )
            fig = px.bar(
                outcome_df, x="Outcome", y="Probability", text="Probability",
                color="Outcome",
            )
            fig.update_traces(texttemplate="%{text:.1%}", showlegend=False)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Goals markets")
            st.write(
                pd.DataFrame(
                    {
                        "Market": [
                            "Over 1.5", "Over 2.5", "Over 3.5",
                            "Both teams to score",
                        ],
                        "Probability": [
                            pct(sim.prob_over(1.5)),
                            pct(sim.prob_over(2.5)),
                            pct(sim.prob_over(3.5)),
                            pct(sim.prob_btts()),
                        ],
                    }
                )
            )

        with right:
            st.subheader("Top 10 exact scores")
            top = sim.top_scores(10).copy()
            top["probability"] = top["probability"].map(lambda p: pct(p))
            top.columns = ["Score", f"{team_a} goals", f"{team_b} goals", "Probability"]
            st.dataframe(top, use_container_width=True, hide_index=True)
            st.caption(
                f"Expected goals: {team_a} {sim.mean_goals_a:.2f} — "
                f"{sim.mean_goals_b:.2f} {team_b}"
            )

# ----------------------------------------------------------- Score Matrix tab ---
with tabs[3]:
    if results is None:
        st.info("Run a single-match simulation first.")
    else:
        sim = results["sim"]
        team_a, team_b = results["team_a"], results["team_b"]
        st.subheader("Exact score probability matrix")
        matrix = sim.score_probability_table(results["max_pred_goals"])
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale="Blues",
                text=[[f"{v:.1%}" for v in row] for row in matrix.values],
                texttemplate="%{text}",
                hovertemplate=(
                    f"{team_a} %{{y}} - %{{x}} {team_b}<br>"
                    "Probability: %{z:.2%}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            xaxis_title=f"{team_b} goals",
            yaxis_title=f"{team_a} goals",
            yaxis_autorange="reversed",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------- Expected Points tab ---
with tabs[4]:
    if results is None:
        st.info("Run a single-match simulation first.")
    else:
        evaluation = results["evaluation"]
        best = best_prediction(evaluation)
        likely = most_likely_score(evaluation)

        st.subheader("Top 10 predictions by expected points")
        top10 = evaluation.head(10).copy()
        display = pd.DataFrame(
            {
                "Prediction": top10["score"],
                "Expected points": top10["expected_points"].round(3),
                "P(exact score)": top10["p_exact"].map(lambda p: pct(p)),
                "P(winner correct)": top10["p_winner_correct"].map(lambda p: pct(p)),
                "P(total goals correct)": top10["p_total_correct"].map(lambda p: pct(p)),
                "P(goal diff correct)": top10["p_diff_correct"].map(lambda p: pct(p)),
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        gap = best["expected_points"] - likely["expected_points"]
        st.caption(
            f"Expected-points edge of {best['score']} over the most likely score "
            f"{likely['score']}: **{gap:+.3f} points per match**."
        )

        st.subheader("Expected points by predicted score")
        chart_df = evaluation.head(25)
        fig = px.bar(
            chart_df, x="score", y="expected_points",
            labels={"score": "Predicted score", "expected_points": "Expected points"},
        )
        fig.update_xaxes(categoryorder="array", categoryarray=chart_df["score"].tolist())
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------ Saved Predictions tab ---
with tabs[5]:
    st.subheader("Saved predictions")
    saved = storage.load_predictions()
    if saved.empty:
        st.info("No saved predictions yet. Run a simulation and click **Save this prediction**.")
    else:
        st.dataframe(saved, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export CSV",
            saved.to_csv(index=False).encode("utf-8"),
            file_name="worldcup_predictions_export.csv",
            mime="text/csv",
        )
        st.caption(
            "⚠️ On Streamlit Community Cloud, this log is wiped whenever the app "
            "restarts or is redeployed. Export the CSV if you want to keep it."
        )

# ----------------------------------------------------------- Methodology tab ---
with tabs[6]:
    st.markdown(
        """
## How this tool works

### Data sources (all automatic)
- **Fixtures**: the full 104-match WC2026 schedule is downloaded from a free
  public feed (fixturedownload.com) — real draw, kickoff times (UTC), venues,
  and live scores. Refresh anytime from the All Matches tab.
- **Odds** (optional but recommended): one click fetches 1X2 and Over/Under
  2.5 odds for every upcoming match from The Odds API (free key, 500
  requests/month — one request covers all matches). The median across
  bookmakers is used.
- **Elo fallback**: when no odds are available, expected goals are derived
  from Elo ratings (`data/elo_ratings.csv`) — zero input required.

### 1. Odds → fair probabilities
Decimal odds are converted to implied probabilities (`1 / odds`). Bookmakers
build in a margin (overround), so the raw probabilities sum to more than 100%.
We remove the margin by normalizing: each probability is divided by the sum.

### 2. Fair probabilities → expected goals
1. **Total goals:** find the total expected goals `λ_total` such that a Poisson
   distribution with that mean gives exactly the market's probability of
   over 2.5 goals (root-finding, bounded between 0.8 and 5.0).
2. **Split between teams:** search over `(λ_A, λ_B)` to minimize the squared
   error between the Poisson model's win/draw/loss/over-2.5 probabilities and
   the market's (SciPy optimization).

### Elo fallback model
When odds are missing: win expectancy `E = 1/(1+10^(−Δelo/400))`, expected
goal margin `Δelo/220` (capped at ±3), total goals `2.5 + 0.25·|margin|`.
Transparent and serviceable, but markets know about injuries and lineups —
prefer odds when available. Details in `model/elo_model.py`.

### 3. Optional transparent adjustments
Small additive xG nudges for injuries, lineups, form, rest, motivation, host
advantage, and Elo/FIFA-ranking gaps. All bounded and documented in
`model/adjustments.py`.

### 4. Monte Carlo simulation
Both teams' goals are drawn from independent Poisson distributions with the
final lambdas, tens of thousands of times.

### 5. Family competition scoring
Every simulated outcome is scored against every candidate prediction
(0-0 up to the configured maximum) using your family rules:

| | Winner/draw correct | Total goals correct | Goal diff correct (winner required) |
|---|---|---|---|
| **Group stage** | 5 | 1 | 3 |
| **Knockout (after 120 min)** | 10 | 2 | 6 |

**Assumption:** the goal-difference bonus is never awarded for draws, because
the rule requires the *winner* to be correct and a draw has no winner
(`DRAW_EARNS_DIFF_BONUS` in `model/scoring.py` if your family rules otherwise).

For knockout matches, predict the score **after 120 minutes** (penalty
shootout excluded) — draws are valid knockout predictions.

### 6. Why the recommendation can differ from the most likely score
The most likely exact score (often 1-1 or 1-0) maximizes only the chance of a
*perfect* hit. But your rules pay mostly for the **winner** and the **goal
difference**. A prediction like 2-1 may have a slightly lower exact-score
probability than 1-1, yet collect winner points in every simulation where the
favorite wins, and goal-difference points in every 1-goal win (1-0, 2-1, 3-2…).
The tool computes the **expected points** of every candidate and recommends
the maximizer.
"""
    )
