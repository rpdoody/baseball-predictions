"""
Entry point for the Rocket Report MLB dashboard.

  - st.set_page_config()  called exactly once here
  - home_page()           landing page with per-game betting recommendations
  - st.navigation()       6-page sidebar navigation
"""

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from page_utils import (
    ROOT,
    _american_to_implied_prob,
    _fetch_espn_odds,
    _fetch_team_standings,
    _fetch_todays_schedule,
    _load_model_results,
    _load_precomputed,
    _prob_bar_html,
    add_betting_oracle_footer,
    init_session_state,
)
from src.models.score_distribution import (
    ScoreDistribution,
    independent_poisson_score_distribution,
)


ET = ZoneInfo("America/New_York")
_LEAGUE_RUNS_PER_GAME = 4.5


def eastern_now() -> datetime.datetime:
    return datetime.datetime.now(ET)


def eastern_today() -> datetime.date:
    return eastern_now().date()


def format_game_time_et(game_datetime: str) -> str:
    """Format an MLB ISO timestamp in US Eastern time, including DST."""
    if not game_datetime:
        return "TBD"
    try:
        dt_utc = datetime.datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
        return dt_utc.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
    except (TypeError, ValueError):
        return "TBD"


def normalize_team_name(team_name: str | None) -> str:
    """Create a comparison-safe representation of team names across feeds."""
    return "".join(char.lower() for char in (team_name or "") if char.isalnum())


def is_matching_odds_game(game: dict, odds_game: dict) -> bool:
    """Match both teams and their designated home/away orientation."""
    return (
        normalize_team_name(game.get("home_name"))
        == normalize_team_name(odds_game.get("home_team"))
        and normalize_team_name(game.get("away_name"))
        == normalize_team_name(odds_game.get("away_team"))
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_todays_schedule(game_date_iso: str):
    """Load the schedule using an explicit Eastern calendar date."""
    game_date = datetime.date.fromisoformat(game_date_iso)
    try:
        return _fetch_todays_schedule(game_date)
    except TypeError:
        # Compatibility fallback while page_utils is updated to accept game_date.
        return _fetch_todays_schedule()


@st.cache_data(ttl=900, show_spinner=False)
def cached_standings():
    return _fetch_team_standings()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_espn_odds():
    return _fetch_espn_odds()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_model_results():
    return _load_model_results()


@st.cache_data(ttl=86400, show_spinner=False)
def cached_precomputed():
    return _load_precomputed()


def _team_run_rate(team_full: str, opponent_full: str, hist_stnd: pd.DataFrame) -> float:
    """Estimate expected runs with offensive/defensive historical team context."""
    try:
        last = team_full.split()[-1]
        team_row = hist_stnd[hist_stnd["team"].str.contains(last, case=False, na=False)]
        opp_last = opponent_full.split()[-1]
        opp_row = hist_stnd[hist_stnd["team"].str.contains(opp_last, case=False, na=False)]
        if not team_row.empty and not opp_row.empty:
            team_row = team_row.sort_values("season").iloc[-1]
            opp_row = opp_row.sort_values("season").iloc[-1]
            offense = float(team_row["RS_per_G"])
            defense = float(opp_row["RA_per_G"])
            blended = (offense * defense) ** 0.5
            return 0.7 * blended + 0.3 * _LEAGUE_RUNS_PER_GAME
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return _LEAGUE_RUNS_PER_GAME


def _coherent_game_distribution(home_full: str, away_full: str, hist_stnd: pd.DataFrame) -> ScoreDistribution:
    """Build a single score distribution used for ML, RL, and totals."""
    home_rate = _team_run_rate(home_full, away_full, hist_stnd)
    away_rate = _team_run_rate(away_full, home_full, hist_stnd)
    return independent_poisson_score_distribution(away_rate, home_rate)


def _short(full_name: str) -> str:
    return full_name.split()[-1] if full_name else ""


def _get_rs_g(team_full: str, hist_stnd: pd.DataFrame) -> float:
    """Most-recent Retrosheet-season RS/G, or league average when unavailable."""
    try:
        last = team_full.split()[-1]
        sub = hist_stnd[hist_stnd["team"].str.contains(last, case=False, na=False)]
        if not sub.empty:
            return float(sub.sort_values("season").iloc[-1]["RS_per_G"])
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return _LEAGUE_RUNS_PER_GAME


def _parse_american(raw) -> int | None:
    """Parse a signed American odds value without fabricating missing data."""
    try:
        value = str(raw).strip().replace("+", "")
        if value.lower() in {"", "—", "none", "nan"}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_american(value: int | None) -> str:
    if value is None:
        return "—"
    return f"+{value}" if value >= 0 else str(value)


def _build_game_recs(g: dict, espn_game: dict | None, standings: dict, hist_stnd: pd.DataFrame) -> dict:
    """Build coherent moneyline, run-line, and total recommendations for one game."""
    del standings  # Retained in the signature for compatibility with callers.

    home_full = g.get("home_name", "")
    away_full = g.get("away_name", "")
    dist = _coherent_game_distribution(home_full, away_full, hist_stnd)
    home_prob = dist.home_moneyline()
    away_prob = 1.0 - home_prob - dist.tie_probability()
    recs: dict = {}

    if not espn_game:
        return recs

    ml_h = _parse_american(espn_game.get("ml_home"))
    ml_a = _parse_american(espn_game.get("ml_away"))
    if ml_h is not None and ml_a is not None:
        impl_h = _american_to_implied_prob(ml_h)
        impl_a = _american_to_implied_prob(ml_a)
        recs["ml"] = {
            "home": {
                "team": home_full,
                "odds_str": _format_american(ml_h),
                "impl": impl_h,
                "est_prob": home_prob,
                "edge": home_prob - impl_h,
            },
            "away": {
                "team": away_full,
                "odds_str": _format_american(ml_a),
                "impl": impl_a,
                "est_prob": away_prob,
                "edge": away_prob - impl_a,
            },
            "best": "home" if home_prob - impl_h >= away_prob - impl_a else "away",
        }

    spread_h = _parse_american(espn_game.get("spread_home"))
    spread_a = _parse_american(espn_game.get("spread_away"))
    if spread_h is not None and spread_a is not None:
        # The lower American moneyline identifies the favorite. If ML prices are
        # absent, infer the -1.5 side from standard run-line price structure.
        if ml_h is not None and ml_a is not None:
            home_favorite = ml_h < ml_a
        else:
            home_favorite = spread_h > 0 and spread_a <= 0

        if home_favorite:
            home_rl, away_rl, _push_rl = dist.run_line_probabilities(-1.5)
            home_pick, away_pick = f"{_short(home_full)} −1.5", f"{_short(away_full)} +1.5"
        else:
            away_rl, home_rl, _push_rl = dist.run_line_probabilities(-1.5)
            home_pick, away_pick = f"{_short(home_full)} +1.5", f"{_short(away_full)} −1.5"

        impl_h = _american_to_implied_prob(spread_h)
        impl_a = _american_to_implied_prob(spread_a)
        recs["rl"] = {
            "home": {
                "pick": home_pick,
                "odds_str": _format_american(spread_h),
                "impl": impl_h,
                "est_prob": home_rl,
                "edge": home_rl - impl_h,
            },
            "away": {
                "pick": away_pick,
                "odds_str": _format_american(spread_a),
                "impl": impl_a,
                "est_prob": away_rl,
                "edge": away_rl - impl_a,
            },
            "best": "home" if home_rl - impl_h >= away_rl - impl_a else "away",
        }

    try:
        posted = float(espn_game.get("over_under"))
    except (TypeError, ValueError):
        posted = None
    over_price = _parse_american(espn_game.get("over_odds"))
    under_price = _parse_american(espn_game.get("under_odds"))

    if posted is not None and over_price is not None and under_price is not None:
        raw_over, raw_under, _push_prob = dist.total_probabilities(posted)
        over_prob = max(0.20, min(0.80, raw_over))
        under_prob = max(0.20, min(0.80, raw_under))
        impl_over = _american_to_implied_prob(over_price)
        impl_under = _american_to_implied_prob(under_price)
        recs["ou"] = {
            "posted": posted,
            "exp_total": _get_rs_g(home_full, hist_stnd) + _get_rs_g(away_full, hist_stnd),
            "over": {
                "pick": f"Over {posted}",
                "odds_str": _format_american(over_price),
                "impl": impl_over,
                "est_prob": over_prob,
                "edge": over_prob - impl_over,
            },
            "under": {
                "pick": f"Under {posted}",
                "odds_str": _format_american(under_price),
                "impl": impl_under,
                "est_prob": under_prob,
                "edge": under_prob - impl_under,
            },
            "best": "over" if over_prob - impl_over >= under_prob - impl_under else "under",
        }

    return recs


def _rec_card_html(label: str, side: dict, exp_info: str) -> str:
    """Render one market recommendation as an HTML block."""
    del label
    edge_pct = side["edge"] * 100
    if edge_pct > 3:
        color, badge = "#16a34a", "✅ BET"
    elif edge_pct > 0:
        color, badge = "#d97706", "➡ LEAN"
    else:
        color, badge = "#dc2626", "⛔ PASS"

    pick_text = _short(side["team"]) if side.get("team") else side.get("pick", "—")
    return (
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:4px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<b style="font-size:0.88rem">{pick_text}</b>'
        f'<span style="background:{color};color:white;border-radius:6px;padding:1px 8px;'
        f'font-size:0.7rem;font-weight:700">{badge}</span></div>'
        f'<div style="font-size:0.78rem;color:#555;margin-top:2px">'
        f"Odds: <b>{side['odds_str']}</b>"
        f' &nbsp;|&nbsp; Edge: <b style="color:{color}">{edge_pct:+.1f}%</b></div>'
        f'<div style="font-size:0.73rem;color:#888">{exp_info}</div>'
        f"</div>"
    )


st.set_page_config(
    page_title="RP Rocket Report - MLB Predictions",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f9fafb; color: #111827; }
    section[data-testid="stSidebar"] { background-color: #001f4d; }
    section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
    h1, h2, h3 { color: #002D72; }
    </style>
    """,
    unsafe_allow_html=True,
)


def home_page() -> None:
    """Landing page: hero metrics and per-game betting recommendations."""
    today_et = eastern_today()

    hdr_left, hdr_right = st.columns([1, 5])
    with hdr_left:
        logo = ROOT / "data_files" / "IMG_0185.PNG"
        if logo.exists():
            st.image(str(logo), width=110)
    with hdr_right:
        st.markdown(
            f"<h1 style='margin-bottom:0;color:#002D72'>RP Rocket Report</h1>"
            f"<p style='color:#6b7280;margin-top:2px'>MLB Predictions &nbsp;·&nbsp; "
            f"{today_et.strftime('%A, %B %d, %Y')}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    games_today = cached_todays_schedule(today_et.isoformat())
    standings = cached_standings()
    espn_odds = cached_espn_odds()
    model_results = cached_model_results()
    precomputed = cached_precomputed()
    hist_stnd = precomputed["standings"]
    init_session_state()

    games_with_odds = sum(
        1 for game in games_today if any(is_matching_odds_game(game, odds) for odds in espn_odds)
    )
    moneyline_metrics = model_results.get("moneyline", {}).get("metrics", {}) if model_results else {}
    accuracy = moneyline_metrics.get("accuracy")
    roc_auc = moneyline_metrics.get("roc_auc")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Today's Games", len(games_today))
    m2.metric("Games with Odds", games_with_odds)
    m3.metric("ML Model AUC", f"{roc_auc:.4f}" if roc_auc is not None else "—", help="Moneyline XGBoost ROC-AUC on held-out test set.")
    m4.metric("Model Accuracy", f"{accuracy:.1%}" if accuracy is not None else "—")
    m5.metric("Odds Source", "ESPN" if espn_odds else "Unavailable")

    st.markdown("---")

    if not games_today:
        st.info("No MLB games scheduled today, or the MLB Stats API is unreachable.")
    else:
        st.markdown("### 🎯 Today's Games & Betting Recommendations")
        st.caption(
            "Win probability, run line, and O/U use one coherent score distribution from each team's "
            "runs-scored/allowed baseline. ✅ BET = edge > 3% · ➡ LEAN = 0–3% · ⛔ PASS = negative edge."
        )

        status_labels = {
            "Final": "🏁 Final",
            "Game Over": "🏁 Final",
            "In Progress": "🔴 LIVE",
            "Scheduled": "🕐 Scheduled",
            "Pre-Game": "⏳ Pre-Game",
            "Warmup": "⏳ Warmup",
            "Delayed": "⚠️ Delayed",
            "Postponed": "🚫 Postponed",
        }

        for idx, game in enumerate(games_today):
            away_full = game.get("away_name", "Away")
            home_full = game.get("home_name", "Home")
            away_sp = game.get("away_probable_pitcher", "TBD") or "TBD"
            home_sp = game.get("home_probable_pitcher", "TBD") or "TBD"
            status = game.get("status", "Scheduled")
            venue = game.get("venue_name", "—")
            game_time = format_game_time_et(game.get("game_datetime", ""))

            score_str = ""
            if str(status).lower() in {"final", "game over", "in progress", "live", "completed"}:
                if game.get("away_score") is not None and game.get("home_score") is not None:
                    score_str = f" &nbsp;·&nbsp; **{game['away_score']}–{game['home_score']}**"

            espn_game = next((odds for odds in espn_odds if is_matching_odds_game(game, odds)), None)
            recs = _build_game_recs(game, espn_game, standings, hist_stnd)
            home_prob = _coherent_game_distribution(home_full, away_full, hist_stnd).home_moneyline()

            with st.container(border=True):
                hdr_c1, hdr_c2 = st.columns([3, 2])
                with hdr_c1:
                    st.markdown(f"#### {away_full} @ {home_full}{score_str}", unsafe_allow_html=True)
                    st.markdown(
                        f"<small>🏟️ {venue} &nbsp;·&nbsp; {status_labels.get(status, status)} "
                        f"&nbsp;·&nbsp; 🕐 {game_time}</small><br>"
                        f"<small>SP: <b>{away_sp}</b> (away) &nbsp;/&nbsp; <b>{home_sp}</b> (home)</small>",
                        unsafe_allow_html=True,
                    )
                with hdr_c2:
                    st.markdown(_prob_bar_html(home_prob, home_full, away_full), unsafe_allow_html=True)

                if not recs:
                    st.caption("⏳ Odds not yet available for this game.")
                    continue

                st.divider()
                col_ml, col_rl, col_ou = st.columns(3)

                with col_ml:
                    st.markdown("##### 💵 Moneyline")
                    if "ml" in recs:
                        ml = recs["ml"]
                        side = ml[ml["best"]]
                        other = ml["away" if ml["best"] == "home" else "home"]
                        exp = f"Est: {side['est_prob']:.0%} · Impl: {side['impl']:.0%}"
                        st.markdown(_rec_card_html("ML", side, exp), unsafe_allow_html=True)
                        st.caption(f"Other side: {_short(other['team'])} {other['odds_str']} (edge {other['edge'] * 100:+.1f}%)")
                    else:
                        st.caption("— odds unavailable —")

                with col_rl:
                    st.markdown("##### 📏 Run Line (±1.5)")
                    if "rl" in recs:
                        rl = recs["rl"]
                        side = rl[rl["best"]]
                        other = rl["away" if rl["best"] == "home" else "home"]
                        exp = f"Est cover: {side['est_prob']:.0%} · Impl: {side['impl']:.0%}"
                        st.markdown(_rec_card_html("RL", side, exp), unsafe_allow_html=True)
                        st.caption(f"Other side: {other['pick']} {other['odds_str']} (edge {other['edge'] * 100:+.1f}%)")
                    else:
                        st.caption("— odds unavailable —")

                with col_ou:
                    st.markdown("##### 📊 Over/Under")
                    if "ou" in recs:
                        ou = recs["ou"]
                        side = ou[ou["best"]]
                        other = ou["under" if ou["best"] == "over" else "over"]
                        exp = f"Exp total: {ou['exp_total']:.1f} · Posted: {ou['posted']} · Impl: {side['impl']:.0%}"
                        st.markdown(_rec_card_html("OU", side, exp), unsafe_allow_html=True)
                        st.caption(f"Other side: {other['pick']} {other['odds_str']} (edge {other['edge'] * 100:+.1f}%)")
                    else:
                        st.caption("— odds unavailable —")

                st.markdown("")
                if st.button("🔍 View Full Game Details →", key=f"home_detail_{idx}", width="stretch"):
                    st.session_state["schedule_selected_game"] = game
                    st.switch_page("pages/1_Today.py")

    st.markdown("---")
    st.markdown("### Explore")
    tiles = [
        ("📅", "Today", "Full schedule with detailed game drill-down", "pages/1_Today.py"),
        ("📊", "Stats", "Standings · Batting · Pitching · Leaders", "pages/2_Stats.py"),
        ("🆚", "Matchup Analysis", "H2H history · Rolling win-rate charts", "pages/3_Matchup_Analysis.py"),
        ("🤖", "Models", "XGBoost features · Evaluation · Savant research", "pages/4_Models.py"),
        ("📈", "Performance", "Pick history · Model P&L · Kelly bankroll", "pages/5_Performance.py"),
        ("ℹ️", "About", "Methodology, data sources & tech stack", "pages/7_Info.py"),
    ]
    for row_tiles in (tiles[:3], tiles[3:]):
        cols = st.columns(3)
        for col, (icon, title, description, path) in zip(cols, row_tiles):
            with col:
                with st.container(border=True):
                    st.markdown(f'<div style="text-align:center;font-size:1.8rem;padding-top:4px">{icon}</div>', unsafe_allow_html=True)
                    st.page_link(path, label=f"**{title}**")
                    st.caption(description)

    add_betting_oracle_footer()


pg = st.navigation(
    {
        "": [
            st.Page(home_page, title="Home", icon="🏠", default=True),
            st.Page("pages/1_Today.py", title="Today", icon="📅"),
            st.Page("pages/2_Stats.py", title="Stats", icon="📊"),
            st.Page("pages/3_Matchup_Analysis.py", title="Matchup Analysis", icon="🆚"),
            st.Page("pages/4_Models.py", title="Models", icon="🤖"),
            st.Page("pages/5_Performance.py", title="Performance", icon="📈"),
            st.Page("pages/6_Pick_6.py", title="Pick 6", icon="🎯"),
            st.Page("pages/7_Info.py", title="About", icon="ℹ️"),
        ]
    }
)
pg.run()