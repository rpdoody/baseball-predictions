from __future__ import annotations

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import statsapi


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrosheet import TEAM_NAMES


ET = ZoneInfo("America/New_York")
SEASON = 2026
OUTPUT_PATH = ROOT / "data_files" / "processed" / f"live_gameinfo_{SEASON}.parquet"

# MLB Stats API team abbreviation -> Retrosheet team code.
MLB_ABBR_TO_RETRO = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN",
    "CWS": "CHA",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "MIA": "FLO",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "ATH": "OAK",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SDN",
    "SEA": "SEA",
    "SF": "SFN",
    "STL": "SLN",
    "TB": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WAS",
}


def retro_code(team_abbr: str) -> str:
    """Translate current MLB abbreviations to Retrosheet-compatible codes."""
    return MLB_ABBR_TO_RETRO.get(str(team_abbr).upper(), str(team_abbr).upper())


def is_completed_game(game: dict) -> bool:
    """Keep only completed regular-season games with final scores."""
    status = str(game.get("status", "")).lower()

    return (
        game.get("game_type") == "R"
        and status in {"final", "game over", "completed"}
        and game.get("away_score") is not None
        and game.get("home_score") is not None
    )


def fetch_completed_games(
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[dict]:
    """Fetch MLB schedule rows and retain completed regular-season games."""
    games = statsapi.schedule(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        sportId=1,
    ) or []

    return [game for game in games if is_completed_game(game)]


def normalize_game(game: dict) -> dict:
    """Normalize an MLB completed-game schedule row to gameinfo-compatible fields."""
    game_date = datetime.date.fromisoformat(game["game_date"])

    away_code = retro_code(game.get("away_abbr", ""))
    home_code = retro_code(game.get("home_abbr", ""))

    away_runs = int(game["away_score"])
    home_runs = int(game["home_score"])

    return {
        "game_id": int(game["game_id"]),
        "gid": f"MLB{game['game_id']}",
        "season": game_date.year,
        "date": int(game_date.strftime("%Y%m%d")),
        "visteam": away_code,
        "hometeam": home_code,
        "vruns": away_runs,
        "hruns": home_runs,
        "wteam": home_code if home_runs > away_runs else away_code,
        "total_runs": away_runs + home_runs,
        "daynight": "",
        "attendance": pd.NA,
        "temp": pd.NA,
        "windspeed": pd.NA,
        "game_type": game.get("game_type", "R"),
        "status": game.get("status", ""),
        "source": "mlb_stats_api",
        "retrieved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def load_existing() -> pd.DataFrame:
    """Load existing output if present, otherwise return an empty compatible frame."""
    if OUTPUT_PATH.exists():
        return pd.read_parquet(OUTPUT_PATH)

    return pd.DataFrame()


def main() -> None:
    today_et = datetime.datetime.now(ET).date()
    last_completed_date = min(today_et - datetime.timedelta(days=1), datetime.date(SEASON, 12, 31))
    season_start = datetime.date(SEASON, 3, 1)

    if last_completed_date < season_start:
        print(f"No completed {SEASON} regular-season games are available yet.")
        return

    print(f"Fetching completed MLB regular-season games: {season_start} through {last_completed_date}")

    games = fetch_completed_games(season_start, last_completed_date)
    rows = [normalize_game(game) for game in games]

    if not rows:
        print("No completed games returned from MLB Stats API.")
        return

    fresh = pd.DataFrame(rows)
    existing = load_existing()

    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = (
        combined.sort_values(["game_id", "retrieved_at_utc"])
        .drop_duplicates(subset=["game_id"], keep="last")
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)

    coverage_dates = pd.to_datetime(combined["date"].astype(str), format="%Y%m%d")

    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Games: {len(combined):,}")
    print(f"First game: {coverage_dates.min().date()}")
    print(f"Last game: {coverage_dates.max().date()}")


if __name__ == "__main__":
    main()