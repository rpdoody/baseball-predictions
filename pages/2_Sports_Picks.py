import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sports Picks", page_icon="⚾", layout="wide")
st.title("⚾ MLB Sports Picks")

path = Path("data_files/best_bets_today.json")

if not path.exists():
    st.info("Today’s bets have not been generated yet.")
    st.stop()

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Could not read the picks export: {exc}")
    st.stop()

meta = payload.get("meta", {})
bets = payload.get("bets", [])
status = meta.get("status", "unknown")

st.caption(
    f"Target date: {meta.get('target_date', 'unknown')} · "
    f"Updated: {meta.get('generated_at', 'unknown')} · "
    f"Status: {status}"
)

if status != "ok":
    st.info(meta.get("notes", "No qualifying bets are available today."))
    st.stop()

df = pd.DataFrame(bets)

if df.empty:
    st.info("No qualifying bets are available today.")
    st.stop()

st.metric("Qualifying bets", len(df))

display_columns = [
    column
    for column in [
        "game_time",
        "game",
        "bet_type",
        "pick",
        "confidence",
        "edge",
        "odds",
        "line",
        "tier",
        "notes",
    ]
    if column in df.columns
]

if "confidence" in df.columns:
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "—"
    )

if "edge" in df.columns:
    df["edge"] = pd.to_numeric(df["edge"], errors="coerce").map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "—"
    )

st.dataframe(
    df[display_columns],
    use_container_width=True,
    hide_index=True,
)
