"""Page: Matchup Analysis — Head-to-Head · Rolling Form"""

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ET = ZoneInfo("America/New_York")

from page_utils import (
    READABLE_COLS,
    _load_precomputed,
    add_betting_oracle_footer,
    render_sidebar,
)
from retrosheet import TEAM_NAMES, head_to_head, rolling_team_form


current_year = datetime.datetime.now(ET).year

min_year, max_year = st.select_slider(
    "Season range",
    options=list(range(2020, current_year + 1)),
    value=(2020, current_year),
    key="season_range_on_page",
)

_pre = _load_precomputed()
_gi = _pre["gameinfo"][_pre["gameinfo"]["season"].between(min_year, max_year)].copy()


def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.

    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)

    Returns:
        int: Calculated height in pixels

    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding

    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height


def _code_to_name(c: str) -> str:
    return TEAM_NAMES.get(str(c).upper(), c)


@st.cache_data(show_spinner=False)
def _cached_head_to_head(team_a: str, team_b: str, start_year: int, end_year: int):
    """Cache head-to-head lookups so toggling between matchups avoids recompute."""
    return head_to_head(team_a, team_b, start_year, end_year)


@st.cache_data(show_spinner=False)
def _cached_rolling_form(team: str, window: int, start_year: int, end_year: int):
    """Cache rolling-form computation so slider/team changes reuse prior results."""
    return rolling_team_form(team, window, start_year, end_year)


teams = sorted(
    set(_gi["visteam"].dropna().map(_code_to_name))
    | set(_gi["hometeam"].dropna().map(_code_to_name))
)

if not teams:
    st.info("No games found for the selected year range. Adjust the sidebar filter.")
    st.stop()


tab_h2h, tab_form = st.tabs(["Head-to-Head", "Rolling Form"])


# ── Head-to-Head ──────────────────────────────────────────────────────────────
with tab_h2h:
    st.subheader("Head-to-Head History")

    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A", teams, index=0, key="h2h_a")
    team_b = col2.selectbox("Team B", teams, index=min(1, len(teams) - 1), key="h2h_b")

    if team_a == team_b:
        st.warning("Select two different teams.")
    else:
        with st.spinner("Loading matchup data…"):
            h2h = _cached_head_to_head(team_a, team_b, min_year, max_year)

        if h2h.empty:
            st.info("No matchups found for the selected teams and year range.")
        else:
            a_wins = int(h2h["a_win"].sum())
            b_wins = len(h2h) - a_wins
            total = len(h2h)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(f"{team_a} wins", f"{a_wins} ({a_wins / total:.1%})")
            mc2.metric(f"{team_b} wins", f"{b_wins} ({b_wins / total:.1%})")
            mc3.metric("Total games", total)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=h2h["date"],
                    y=h2h["a_runs"],
                    mode="markers+lines",
                    name=f"{team_a} runs",
                    line=dict(color="#1f77b4"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=h2h["date"],
                    y=h2h["b_runs"],
                    mode="markers+lines",
                    name=f"{team_b} runs",
                    line=dict(color="#d62728"),
                )
            )
            fig.update_layout(
                title=f"{team_a} vs {team_b} — Runs by game",
                xaxis_title="Date",
                yaxis_title="Runs",
            )
            st.plotly_chart(fig, width="stretch")

            h2h["season"] = h2h["date"].dt.year
            by_season = (
                h2h.groupby("season")
                .agg(
                    a_wins=("a_win", "sum"),
                    games=("a_win", "count"),
                )
                .reset_index()
            )
            by_season["a_wpct"] = by_season["a_wins"] / by_season["games"]

            fig2 = px.bar(
                by_season,
                x="season",
                y="a_wpct",
                title=f"{team_a} win % vs {team_b} by season",
                labels={"a_wpct": f"{team_a} W%", "season": "Season"},
                color="a_wpct",
                color_continuous_scale="RdYlGn",
            )
            fig2.add_hline(y=0.5, line_dash="dot", line_color="gray")
            fig2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig2, width="stretch")

            with st.expander("Raw game log"):
                log_df = (
                    h2h[["date", "visteam", "hometeam", "vruns", "hruns", "a_win"]]
                    .assign(date=lambda d: d["date"].dt.date)
                    .rename(columns={**READABLE_COLS, "a_win": f"{team_a} Win"})
                )
                st.dataframe(
                    log_df,
                    width="stretch",
                    hide_index=True,
                    height=get_dataframe_height(log_df),
                )


# ── Rolling Form ──────────────────────────────────────────────────────────────
with tab_form:
    st.subheader("Team Rolling Form")

    form_team = st.selectbox("Team", teams, key="form_team")
    form_window = st.slider("Rolling window (games)", 5, 30, 10, key="form_window")

    with st.spinner(f"Computing {form_window}-game rolling form for {form_team}…"):
        form_df = _cached_rolling_form(form_team, form_window, min_year, max_year)

    if form_df.empty:
        st.info("No data found.")
    else:
        rs_col = f"roll_RS_{form_window}"
        ra_col = f"roll_RA_{form_window}"
        rw_col = f"roll_W_{form_window}"
        required_cols = {rs_col, ra_col, rw_col, "roll_RD"}
        missing_cols = required_cols - set(form_df.columns)

        if missing_cols:
            st.warning(
                f"Rolling window of {form_window} games produced incomplete data "
                f"(missing: {', '.join(sorted(missing_cols))}). Try a smaller window."
            )
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=form_df["date"], y=form_df[rs_col], name="Avg RS", line=dict(color="green")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=form_df["date"], y=form_df[ra_col], name="Avg RA", line=dict(color="red")
                )
            )
            fig.update_layout(
                title=f"{form_team} — {form_window}-game rolling runs scored/allowed",
                xaxis_title="Date",
                yaxis_title="Runs (rolling avg)",
            )
            st.plotly_chart(fig, width="stretch")

            fig2 = px.line(
                form_df,
                x="date",
                y=rw_col,
                title=f"{form_team} — {form_window}-game rolling win rate",
                labels={rw_col: "Win rate", "date": "Date"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig2.add_hline(y=0.5, line_dash="dot", line_color="gray")
            st.plotly_chart(fig2, width="stretch")

            st.markdown("#### Recent 20 games")
            recent_20 = (
                form_df.tail(20)[["date", "RS", "RA", "W", rs_col, ra_col, rw_col, "roll_RD"]]
                .sort_values("date", ascending=False)
                .reset_index(drop=True)
                .assign(date=lambda d: d["date"].dt.date)
                .rename(
                    columns={
                        **READABLE_COLS,
                        rs_col: f"Avg RS ({form_window}g)",
                        ra_col: f"Avg RA ({form_window}g)",
                        rw_col: f"Win Rate ({form_window}g)",
                        "roll_RD": "Rolling Run Diff",
                    }
                )
            )
            st.dataframe(
                recent_20,
                width="stretch",
                hide_index=True,
                height=get_dataframe_height(recent_20, max_height=760),
            )
