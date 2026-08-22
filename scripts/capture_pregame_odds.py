import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_files" / "processed" / "pregame_odds_snapshots.parquet"
ET = ZoneInfo("America/New_York")


def get_api_key() -> str:
    api_key = os.getenv("ODDS_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st
        return st.secrets["ODDS_API_KEY"]
    except Exception as exc:
        raise RuntimeError(
            "ODDS_API_KEY is missing. Add it to .streamlit/secrets.toml "
            "or export it in the terminal before running this script."
        ) from exc


def fetch_odds(api_key: str) -> list[dict]:
    response = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
        timeout=30,
    )
    response.raise_for_status()

    print(
        "API requests remaining:",
        response.headers.get("x-requests-remaining", "unknown"),
    )
    return response.json()


def main() -> None:
    api_key = get_api_key()
    captured_at_utc = datetime.datetime.now(datetime.timezone.utc)
    captured_at_et = captured_at_utc.astimezone(ET)

    rows: list[dict] = []

    for event in fetch_odds(api_key):
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "captured_at_utc": captured_at_utc,
                            "captured_at_et": captured_at_et.isoformat(),
                            "game_date_et": captured_at_et.date().isoformat(),
                            "event_id": event.get("id"),
                            "commence_time_utc": event.get("commence_time"),
                            "away_team": event.get("away_team"),
                            "home_team": event.get("home_team"),
                            "bookmaker": bookmaker.get("key"),
                            "market": market.get("key"),
                            "outcome_name": outcome.get("name"),
                            "outcome_price": outcome.get("price"),
                            "outcome_point": outcome.get("point"),
                            "source": "the_odds_api",
                        }
                    )

    new_df = pd.DataFrame(rows)
    if new_df.empty:
        raise RuntimeError("The Odds API returned no MLB odds rows.")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    if OUT.exists():
        old_df = pd.read_parquet(OUT)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[
                "captured_at_utc",
                "event_id",
                "bookmaker",
                "market",
                "outcome_name",
                "outcome_point",
            ],
            keep="last",
        )
    else:
        combined = new_df

    combined.to_parquet(OUT, index=False)

    print(f"Captured {len(new_df):,} odds rows.")
    print(f"Snapshot file: {OUT}")
    print(f"Total saved rows: {len(combined):,}")


if __name__ == "__main__":
    main()
