"""Page: Today — today's schedule, pitcher matchups & odds detail."""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from page_utils import (
    _MLB_TO_RETRO,
    READABLE_COLS,
    ROOT,
    _fetch_espn_odds,
    _fetch_pitcher_stats,
    _fetch_pitcher_throw_hand,
    _fetch_team_il_players,
    _fetch_team_rest_days,
    _fetch_team_standings,
    _fetch_todays_schedule,
    _load_game_context_cache,
    _load_latest_odds,
    _fetch_confirmed_lineups,
    init_session_state,
    render_sidebar,
)
from retrosheet import head_to_head, load_gameinfo, rolling_team_form
from src.ingestion.weather import fetch_forecast


ET = ZoneInfo("America/New_York")


def eastern_now() -> datetime.datetime:
    return datetime.datetime.now(ET)


def eastern_today() -> datetime.date:
    return eastern_now().date()


def format_game_time_et(game_datetime: str) -> str:
    """Convert MLB ISO UTC datetime to DST-safe US Eastern display time."""
    if not game_datetime:
        return "TBD"
    try:
        dt_utc = datetime.datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
        return dt_utc.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
    except (TypeError, ValueError):
        return "TBD"


def game_date_from_datetime(game_datetime: str) -> str:
    """Return the game date in Eastern time, not UTC calendar time."""
    if not game_datetime:
        return ""
    try:
        dt_utc = datetime.datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
        return dt_utc.astimezone(ET).date().isoformat()
    except (TypeError, ValueError):
        return ""


def normalize_team_name(team_name: str | None) -> str:
    """Normalize team names for stable comparisons across MLB and ESPN feeds."""
    return "".join(char.lower() for char in (team_name or "") if char.isalnum())


def is_matching_odds_game(game: dict, odds_game: dict) -> bool:
    """Require both teams and their away/home orientation to match."""
    return normalize_team_name(game.get("away_name")) == normalize_team_name(
        odds_game.get("away_team")
    ) and normalize_team_name(game.get("home_name")) == normalize_team_name(
        odds_game.get("home_team")
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_todays_schedule(game_date_iso: str):
    """Short-lived cache for a live schedule; use the explicit ET date."""
    game_date = datetime.date.fromisoformat(game_date_iso)
    try:
        return _fetch_todays_schedule(game_date)
    except TypeError:
        # Backward compatibility while the helper is being updated.
        return _fetch_todays_schedule()


@st.cache_data(ttl=900, show_spinner=False)
def cached_standings():
    return _fetch_team_standings()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_pitcher_stats(pitcher_name: str):
    return _fetch_pitcher_stats(pitcher_name)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_pitcher_hand(pitcher_name: str):
    return _fetch_pitcher_throw_hand(pitcher_name)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_team_rest_days(team_name: str):
    return _fetch_team_rest_days(team_name)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_team_il_players(team_name: str):
    return _fetch_team_il_players(team_name)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_espn_odds():
    return _fetch_espn_odds()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_weather(venue_name: str, game_date_iso: str):
    return fetch_forecast(venue_name, game_date_iso)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_h2h(away_team: str, home_team: str, start_year: int, end_year: int):
    return head_to_head(away_team, home_team, start_year, end_year)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_rolling_form(team: str, window: int, start_year: int, end_year: int):
    return rolling_team_form(team, window, start_year, end_year)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_gameinfo():
    return load_gameinfo()[["gid", "hometeam", "season"]]


def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    calculated_height = (len(df) * row_height) + header_height + padding
    return min(calculated_height, max_height) if max_height is not None else calculated_height


def format_flag_value(val):
    if val is True:
        return "Yes ✅"
    if val is False:
        return "No ❌"
    return "N/A"


init_session_state()
render_sidebar(show_year_filter=False)

_today_et = eastern_today()
_games_today = cached_todays_schedule(_today_et.isoformat())


# ── Game Detail View ──────────────────────────────────────────────────────────
if st.session_state["schedule_selected_game"] is not None:
    g = st.session_state["schedule_selected_game"]
    away_full = g.get("away_name", "Away")
    home_full = g.get("home_name", "Home")
    away_retro = _MLB_TO_RETRO.get(away_full, away_full)
    home_retro = _MLB_TO_RETRO.get(home_full, home_full)

    if st.button("← Back to Schedule", key="back_to_schedule"):
        st.session_state["schedule_selected_game"] = None
        st.rerun()

    st.markdown(f"## {away_full} @ {home_full}")

    status = g.get("status", "Scheduled")
    venue = g.get("venue_name", "—")
    series = g.get("series_description", "")
    gtime_raw = g.get("game_datetime", "")
    gtime_str = format_game_time_et(gtime_raw)

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Status", status)
    dc2.metric("Game Time", gtime_str)
    dc3.metric("Venue", venue)
    dc4.metric("Series", series or "Regular Season")
    st.divider()

    _standings = cached_standings()
    away_rec = _standings.get(away_full, {})
    home_rec = _standings.get(home_full, {})
    if away_rec or home_rec:
        st.markdown("### 📋 Team Records (Current Season)")
        tr1, tr2 = st.columns(2)
        for col, team_name, team_rec in [(tr1, away_full, away_rec), (tr2, home_full, home_rec)]:
            with col:
                side = "Away" if team_name == away_full else "Home"
                st.markdown(f"**{team_name} ({side})**")
                if team_rec:
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("W-L", f"{team_rec['W']}-{team_rec['L']}")
                    rc2.metric("Win %", team_rec.get("pct", "—"))
                    rc3.metric("Streak", team_rec.get("streak", "—"))
                    rc4.metric("Last 10", team_rec.get("L10", "—"))
                else:
                    st.caption("Record unavailable.")
        st.divider()

    away_sp = g.get("away_probable_pitcher", "TBD") or "TBD"
    home_sp = g.get("home_probable_pitcher", "TBD") or "TBD"
    st.markdown("### ⚾ Probable Pitchers")
    with st.spinner("Fetching pitcher stats…"):
        away_sp_stats = cached_pitcher_stats(away_sp)
        home_sp_stats = cached_pitcher_stats(home_sp)

    pc1, pc2 = st.columns(2)
    for col, team_name, side, pitcher, pitcher_stats in [
        (pc1, away_full, "Away", away_sp, away_sp_stats),
        (pc2, home_full, "Home", home_sp, home_sp_stats),
    ]:
        with col:
            st.markdown(f"**{team_name} ({side})** · {pitcher}")
            if pitcher_stats:
                st.dataframe(
                    pd.DataFrame(pitcher_stats.items(), columns=["Stat", "Value"]),
                    hide_index=True,
                    width="stretch",
                )
            elif pitcher != "TBD":
                st.caption("Stats not yet available for this season.")

    st.divider()
    st.markdown("### 📋 Confirmed Lineups")

    _game_pk = g.get("game_id") or g.get("game_pk")
    _lineups = _fetch_confirmed_lineups(_game_pk)

    if not _lineups["confirmed"]:
        st.caption(
            "Official MLB.com lineups have not been confirmed yet. "
            "Check back closer to first pitch."
        )
    else:
        _lu1, _lu2 = st.columns(2)

        with _lu1:
            st.markdown(f"**{away_full} batting order**")
            st.dataframe(
                pd.DataFrame(_lineups["away"]).rename(
                    columns={
                        "order": "#",
                        "player": "Player",
                        "position": "Pos",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        with _lu2:
            st.markdown(f"**{home_full} batting order**")
            st.dataframe(
                pd.DataFrame(_lineups["home"]).rename(
                    columns={
                        "order": "#",
                        "player": "Player",
                        "position": "Pos",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

    st.divider()
    st.markdown("### 🔍 Game Context Factors")
    _ctx = _load_game_context_cache()
    _pf = _ctx["park_factors"].get(home_retro)
    _park_runs_avg = _ctx["ump_park_avg"].get(home_retro)
    _dn_data = _ctx["daynight"]

    _pf_label = f"{_pf:.3f}" if _pf is not None else "N/A"
    _pf_delta = ""
    if _pf is not None:
        _pf_delta = (
            "↑ hitter-friendly"
            if _pf > 1.05
            else ("↓ pitcher-friendly" if _pf < 0.95 else "≈ neutral")
        )
    _park_runs_label = f"{_park_runs_avg:.1f} R/G" if _park_runs_avg is not None else "N/A"

    _ctx_v1, _ctx_v2, _ctx_v3 = st.columns(3)
    _ctx_v1.metric(
        "🏟️ Park Factor",
        _pf_label,
        delta=_pf_delta or None,
        delta_color="normal",
        help="Average total runs/game at this park versus league average (>1.0 = hitter-friendly). Last 3 seasons.",
    )
    _ctx_v2.metric(
        "🏟️ Park Historical Runs/G",
        _park_runs_label,
        help="Historical average total runs per game at this home park.",
    )

    _home_sp_hand = cached_pitcher_hand(home_sp)
    _away_sp_hand = cached_pitcher_hand(away_sp)
    _home_plat = _ctx["platoon"].get(home_retro, {})
    _away_plat = _ctx["platoon"].get(away_retro, {})

    def plat_adv(bat_pct_left: float, sp_throws: str) -> str:
        if sp_throws == "L":
            return f"{1 - bat_pct_left:.0%} RHB vs LHP"
        if sp_throws == "R":
            return f"{bat_pct_left:.0%} LHB vs RHP"
        return "?"

    home_bat_adv = plat_adv(_home_plat.get("pct_left", 0.5), _away_sp_hand)
    away_bat_adv = plat_adv(_away_plat.get("pct_left", 0.5), _home_sp_hand)
    with _ctx_v3:
        st.markdown("**⚔️ Platoon Matchup**")
        away_last_name = away_sp.split()[-1] if away_sp != "TBD" else "TBD"
        home_last_name = home_sp.split()[-1] if home_sp != "TBD" else "TBD"
        st.caption(
            f"Away SP: **{away_last_name} ({_away_sp_hand}HP)** → Home batters: {home_bat_adv}"
        )
        st.caption(
            f"Home SP: **{home_last_name} ({_home_sp_hand}HP)** → Away batters: {away_bat_adv}"
        )

    st.divider()
    gc_away, gc_home = st.columns(2)
    for col, team_full, team_retro, is_away in [
        (gc_away, away_full, away_retro, True),
        (gc_home, home_full, home_retro, False),
    ]:
        with col:
            st.markdown(f"**{team_full} ({'Away' if is_away else 'Home'})**")
            rest = cached_team_rest_days(team_full)
            if rest is None:
                rest_label, rest_help = "N/A", "Could not determine — check schedule."
            elif rest == 0:
                rest_label, rest_help = "Back-to-back", "Played yesterday."
            else:
                rest_label, rest_help = f"{rest}d rest", f"{rest} day(s) since last game."

            bp_ipg = _ctx["bullpen_ip_pg"].get(team_retro)
            bp_label = f"{bp_ipg:.1f} IP/G" if bp_ipg is not None else "N/A"
            dn_team = _dn_data.get(team_retro, {})
            day_w, night_w = dn_team.get("day"), dn_team.get("night")
            dn_label = (
                f"Day {day_w:.0%} / Night {night_w:.0%}"
                if day_w is not None and night_w is not None
                else "N/A"
            )

            r1c1, r1c2 = col.columns(2)
            r1c1.metric("📅 Rest", rest_label, help=rest_help)
            r1c2.metric(
                "💪 Bullpen IP/G",
                bp_label,
                help="Average relief innings/game over the last two seasons.",
            )
            col.metric(
                "🌙 Day/Night W%",
                dn_label,
                help="Historical win percentage in day versus night games over the last three seasons.",
            )

            il_players = cached_team_il_players(team_full)
            if il_players:
                with col.expander(
                    f"🏥 IL ({len(il_players)} player{'s' if len(il_players) != 1 else ''})"
                ):
                    for player_name in sorted(il_players):
                        st.markdown(f"- {player_name}")
            else:
                col.caption("🏥 IL: None reported")

    st.divider()
    cur_year = _today_et.year
    st.markdown("### 🆚 Head-to-Head History (2020–present)")
    with st.spinner("Loading H2H data…"):
        h2h_detail = cached_h2h(away_retro, home_retro, 2020, cur_year)

    if h2h_detail.empty:
        st.info(
            f"No historical matchups found between **{away_retro}** and **{home_retro}** in the 2020–{cur_year} dataset."
        )
    else:
        away_wins = int(h2h_detail["a_win"].sum())
        home_wins = len(h2h_detail) - away_wins
        total_games = len(h2h_detail)
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric(f"{away_retro} wins", f"{away_wins} ({away_wins / total_games:.0%})")
        hc2.metric(f"{home_retro} wins", f"{home_wins} ({home_wins / total_games:.0%})")
        hc3.metric("Games played", total_games)

        fig_h2h = go.Figure()
        fig_h2h.add_trace(
            go.Scatter(
                x=h2h_detail["date"],
                y=h2h_detail["a_runs"],
                mode="markers+lines",
                name=f"{away_retro} runs",
                line=dict(color="#1f77b4"),
            )
        )
        fig_h2h.add_trace(
            go.Scatter(
                x=h2h_detail["date"],
                y=h2h_detail["b_runs"],
                mode="markers+lines",
                name=f"{home_retro} runs",
                line=dict(color="#d62728"),
            )
        )
        fig_h2h.update_layout(
            title=f"{away_retro} vs {home_retro} — Runs per game",
            xaxis_title="Date",
            yaxis_title="Runs",
        )
        st.plotly_chart(fig_h2h, width="stretch")

        by_szn = (
            h2h_detail.assign(season=h2h_detail["date"].dt.year)
            .groupby("season")
            .agg(away_wins=("a_win", "sum"), games=("a_win", "count"))
            .reset_index()
        )
        by_szn["away_wpct"] = by_szn["away_wins"] / by_szn["games"]
        fig_szn = px.bar(
            by_szn,
            x="season",
            y="away_wpct",
            title=f"{away_retro} win % vs {home_retro} by season",
            labels={"away_wpct": f"{away_retro} W%", "season": "Season"},
            color="away_wpct",
            color_continuous_scale="RdYlGn",
        )
        fig_szn.add_hline(y=0.5, line_dash="dot", line_color="gray")
        fig_szn.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_szn, width="stretch")

        with st.expander("Full game log"):
            st.dataframe(
                h2h_detail[["date", "visteam", "hometeam", "vruns", "hruns", "a_win"]]
                .assign(date=lambda d: d["date"].dt.date)
                .rename(columns={**READABLE_COLS, "a_win": f"{away_retro} Win"}),
                hide_index=True,
                width="stretch",
            )

    st.divider()
    st.markdown("### 📈 Recent Form — Last 20 Games")
    fc1, fc2 = st.columns(2)
    for col, team_retro, team_full, color in [
        (fc1, away_retro, away_full, "#1f77b4"),
        (fc2, home_retro, home_full, "#d62728"),
    ]:
        with col:
            st.markdown(f"**{team_full}**")
            with st.spinner(f"Loading {team_retro} form…"):
                form = cached_rolling_form(team_retro, 10, 2020, cur_year)
            if form.empty:
                st.caption("No form data available.")
            else:
                recent = form.tail(20)
                fig_form = px.line(
                    recent,
                    x="date",
                    y="roll_W_10",
                    title=f"{team_retro} — 10-game win rate",
                    labels={"roll_W_10": "Win rate (10g)", "date": ""},
                    color_discrete_sequence=[color],
                )
                fig_form.add_hline(y=0.5, line_dash="dot", line_color="gray")
                fig_form.update_layout(height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_form, width="stretch")
                last5 = form.tail(5)[["date", "RS", "RA", "W"]].copy()
                last5["date"] = last5["date"].dt.strftime("%b %d")
                last5["Result"] = last5["W"].map({1: "✔ W", 0: "✘ L"})
                st.dataframe(
                    last5[["date", "RS", "RA", "Result"]].rename(
                        columns={"date": "Date", "RS": "R"}
                    ),
                    hide_index=True,
                )

    st.divider()
    st.markdown("### 💰 Odds")
    espn_odds_list = cached_espn_odds()
    game_espn = next((eo for eo in espn_odds_list if is_matching_odds_game(g, eo)), None)

    if game_espn:
        st.caption(
            f"Source: **{game_espn['provider']}** (ESPN public API) · refreshes every 30 min"
        )
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.markdown("**💵 Moneyline**")
            st.metric(away_full, str(game_espn["ml_away"]))
            st.metric(home_full, str(game_espn["ml_home"]))
        with oc2:
            st.markdown("**📏 Run Line**")
            st.metric("Spread", str(game_espn["details"]))
            st.metric("Home spread odds", str(game_espn["spread_home"]))
        with oc3:
            st.markdown("**📊 Over/Under**")
            st.metric("Total", str(game_espn["over_under"]))
            st.markdown(
                f"Over: **{game_espn['over_odds']}** &nbsp;/&nbsp; Under: **{game_espn['under_odds']}**",
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "No ESPN odds found for this game yet. Odds typically open 1–2 days before game time."
        )

    odds_csv = _load_latest_odds()
    st.markdown("### 🌤️ Weather & Situational Context")
    venue_name = g.get("venue_name", "")
    game_date_str = game_date_from_datetime(gtime_raw)
    wx = cached_weather(venue_name, game_date_str) if venue_name and game_date_str else None

    if wx is None:
        st.info("⚠️ Weather data unavailable — venue not recognised or network error.")
    elif wx.get("is_dome"):
        st.info(
            f"🏟️ **{venue_name}** has a retractable / fixed roof — weather has no significant impact on this game."
        )
        st.markdown("**Model Weather Flags**")
        dome_flag_defs = [
            ("Wind Out", None, "Not applicable for domed parks"),
            ("Wind In", None, "Not applicable for domed parks"),
            ("Dome Park", True, "Enclosed/retractable roof park — weather effects are muted"),
            ("Cold Temp", None, "Not applicable for domed parks"),
            ("Hot Temp", None, "Not applicable for domed parks"),
            ("Overcast", None, "Not applicable for domed parks"),
        ]
        cols = st.columns(3)
        for i, (name, value, help_text) in enumerate(dome_flag_defs):
            cols[i % 3].metric(name, format_flag_value(value), help=help_text)
    else:
        temp_f = wx.get("temp_f", 0.0)
        wind_mph = wx.get("wind_mph", 0.0)
        precip = wx.get("precip_mm", 0.0)
        humid = wx.get("humidity_pct", 0.0)
        cloud = wx.get("cloud_cover_pct", 0.0)
        is_past = game_date_str < _today_et.isoformat()
        wx_api = "archive" if is_past else "forecast"
        cols = st.columns(4)
        cols[0].metric("Temperature", f"{temp_f:.0f} °F")
        cols[1].metric("Wind", f"{wind_mph:.0f} mph")
        cols[2].metric("Precip", f"{precip:.1f} mm")
        cols[3].metric("Cloud Cover", f"{cloud:.0f}%")
        st.caption(
            f"Humidity: {humid:.0f}% · Source: Open-Meteo {wx_api} at **{venue_name}** · Game-time hours averaged (1–9 PM local)"
        )

        st.markdown("**Model Weather Flags**")
        flag_defs = [
            (
                "Wind Out",
                None,
                "Wind blowing toward outfield (park-specific). Not modeled directly.",
            ),
            (
                "Wind In",
                None,
                "Wind blowing into the infield (park-specific). Not modeled directly.",
            ),
            ("Dome Park", False, "Outdoor park — dome flag is off"),
            ("Cold Temp", temp_f < 50, "Temp < 50 °F — offense may be suppressed"),
            ("Hot Temp", temp_f > 90, "Temp > 90 °F — offense may be elevated"),
            ("Overcast", cloud > 75, "Cloud cover > 75% — overcast conditions"),
        ]
        cols = st.columns(3)
        for i, (name, value, help_text) in enumerate(flag_defs):
            cols[i % 3].metric(name, format_flag_value(value), help=help_text)

    wx_hist_path = ROOT / "data_files" / "processed" / "weather_historical.parquet"
    with st.expander("📅 Historical Weather at This Venue"):
        if not wx_hist_path.exists():
            st.info(
                "Historical weather data has not been built yet. Run `python scripts/fetch_weather_history.py` once to populate it."
            )
        else:
            try:
                wx_hist_df = pd.read_parquet(wx_hist_path)
                venue_wx = wx_hist_df.merge(cached_gameinfo(), on="gid", how="left")
                venue_wx = venue_wx[venue_wx["hometeam"] == home_retro]
                if venue_wx.empty:
                    st.caption("No historical weather rows found for this home team/venue.")
                else:
                    num_cols = [
                        c
                        for c in [
                            "temp_f",
                            "wind_mph",
                            "precip_mm",
                            "humidity_pct",
                            "cloud_cover_pct",
                        ]
                        if c in venue_wx.columns
                    ]
                    by_season = (
                        venue_wx.groupby("season")
                        .agg(Games=("gid", "count"), **{c: (c, "mean") for c in num_cols})
                        .reset_index()
                        .sort_values("season", ascending=False)
                    )
                    rename = {
                        "season": "Season",
                        "temp_f": "Avg Temp (°F)",
                        "wind_mph": "Avg Wind (mph)",
                        "precip_mm": "Avg Precip (mm)",
                        "humidity_pct": "Avg Humidity (%)",
                        "cloud_cover_pct": "Avg Cloud (%)",
                    }
                    st.dataframe(
                        by_season.rename(columns=rename)
                        .round(
                            {
                                "Avg Temp (°F)": 1,
                                "Avg Wind (mph)": 1,
                                "Avg Precip (mm)": 2,
                                "Avg Humidity (%)": 1,
                                "Avg Cloud (%)": 1,
                            }
                        )
                        .reset_index(drop=True),
                        hide_index=True,
                        width="stretch",
                    )
                    st.caption(
                        f"Averages across game-time hours (1–9 PM local) · {len(venue_wx):,} games · {home_retro} home park"
                    )
            except Exception as exc:
                st.warning(f"Could not load historical weather: {exc}")

    if not odds_csv.empty:
        with st.expander("📚 Multi-book comparison (Odds API)"):
            import os

            has_key = bool(os.environ.get("ODDS_API_KEY"))
            st.caption(
                "Live odds fetched automatically from The Odds API (refreshed hourly)."
                if has_key
                else "Showing saved odds data. Set `ODDS_API_KEY` in the .env file to enable automatic live fetching."
            )
            home_norm = normalize_team_name(home_full)
            away_norm = normalize_team_name(away_full)
            game_odds = odds_csv[
                odds_csv.apply(
                    lambda row: (
                        normalize_team_name(row.get("home_team")) == home_norm
                        and normalize_team_name(row.get("away_team")) == away_norm
                    ),
                    axis=1,
                )
            ].copy()
            if game_odds.empty:
                st.caption("No multi-book data for this matchup in the saved file.")
            else:
                for market_key, market_label in [
                    ("h2h", "💵 Moneyline"),
                    ("spreads", "📏 Run Line"),
                    ("totals", "📊 Over/Under"),
                ]:
                    market_df = game_odds[game_odds["market"] == market_key]
                    if not market_df.empty:
                        st.markdown(f"**{market_label}**")
                        st.dataframe(
                            market_df[
                                ["bookmaker", "outcome_name", "outcome_price", "outcome_point"]
                            ]
                            .sort_values("bookmaker")
                            .rename(
                                columns={
                                    "bookmaker": "Book",
                                    "outcome_name": "Side",
                                    "outcome_price": "Odds",
                                    "outcome_point": "Line",
                                }
                            ),
                            hide_index=True,
                            width="stretch",
                        )


# ── Schedule List View ────────────────────────────────────────────────────────
else:
    today_str = _today_et.strftime("%A, %B %d, %Y")
    st.subheader(f"Today's Schedule — {today_str}")

    if not _games_today:
        st.info(
            "No MLB games scheduled today, or the MLB Stats API is unreachable. Check back on a game day."
        )
    else:
        st.caption(f"{len(_games_today)} game{'s' if len(_games_today) != 1 else ''} today")
        status_badge = {
            "Final": "🏁",
            "Game Over": "🏁",
            "In Progress": "🔴 LIVE",
            "Scheduled": "🕐",
            "Pre-Game": "⏳",
            "Warmup": "⏳",
            "Delayed": "⚠️",
            "Suspended": "⚠️",
            "Postponed": "🚫",
            "Cancelled": "🚫",
        }

        for idx, game in enumerate(_games_today):
            away_name = game.get("away_name", "Away")
            home_name = game.get("home_name", "Home")
            away_sp = game.get("away_probable_pitcher", "TBD") or "TBD"
            home_sp = game.get("home_probable_pitcher", "TBD") or "TBD"
            venue = game.get("venue_name", "—")
            status = game.get("status", "Scheduled")
            status_icon = status_badge.get(status, "")
            gtime_str = format_game_time_et(game.get("game_datetime", ""))
            score_str = ""
            if (
                str(status).lower() in {"final", "game over", "in progress", "live", "completed"}
                and game.get("away_score") is not None
                and game.get("home_score") is not None
            ):
                score_str = f"  **{game['away_score']} – {game['home_score']}**"

            with st.container(border=True):
                sc1, sc2, sc3 = st.columns([4, 3, 2])
                with sc1:
                    st.markdown(
                        f"**{away_name}** @ **{home_name}**{score_str}  \n<small>🏟 {venue}</small>",
                        unsafe_allow_html=True,
                    )
                with sc2:
                    st.markdown(
                        f"<small>🕐 {gtime_str} &nbsp;|&nbsp; {status_icon} {status}</small>  \n<small>Away SP: {away_sp}</small>  \n<small>Home SP: {home_sp}</small>",
                        unsafe_allow_html=True,
                    )
                with sc3:
                    if st.button("View Details →", key=f"sched_detail_{idx}", width="stretch"):
                        st.session_state["schedule_selected_game"] = game
                        st.rerun()
