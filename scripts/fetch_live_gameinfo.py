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
    "KCR": "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "ATH": "OAK",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SDN",
    "SDP": "SDN",
    "SEA": "SEA",
    "SF": "SFN",
    "SFG": "SFN",
    "STL": "SLN",
    "TB": "TBA",
    "TBR": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WAS",
    "WAS": "WAS",
}
MLB_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Angels of Anaheim": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SF",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def retro_code(team_abbr: str) -> str:
    """Translate MLB Stats API abbreviations to Retrosheet team codes."""
    return MLB_ABBR_TO_RETRO.get(
        str(team_abbr).strip().upper(),
        str(team_abbr).strip().upper(),
    )


def is_completed_game(game: dict) -> bool:
    """Keep only completed regular-season games with final scores."""
    status = str(game.get("status", "")).strip().lower()

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
    """Fetch regular-season schedule games and retain completed games."""
    games = (
        statsapi.schedule(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            sportId=1,
        )
        or []
    )

    return [game for game in games if is_completed_game(game)]


def _schedule_team_abbr(game: dict, side: str) -> str:
    """
    Resolve a team's MLB abbreviation from a statsapi.schedule() game record.

    `side` must be either 'away' or 'home'.
    """
    direct_keys = (
        f"{side}_abbr",
        f"{side}_abbreviation",
        f"{side}_team_abbr",
    )

    for key in direct_keys:
        value = game.get(key)
        if value:
            return str(value).upper()

    name_keys = (
        f"{side}_name",
        f"{side}_team_name",
    )

    for key in name_keys:
        value = game.get(key)
        if value:
            abbreviation = MLB_NAME_TO_ABBR.get(str(value).strip())
            if abbreviation:
                return abbreviation

    return ""


def normalize_game(game: dict) -> dict:
    """Normalize an MLB completed-game schedule row to gameinfo-compatible fields."""
    game_date = datetime.date.fromisoformat(game["game_date"])

    away_abbr = _schedule_team_abbr(game, "away")
    home_abbr = _schedule_team_abbr(game, "home")

    if not away_abbr or not home_abbr:
        raise ValueError(
            "Unable to determine MLB team abbreviations for game "
            f"{game.get('game_id')}. Available keys: {sorted(game.keys())}"
        )

    away_code = retro_code(away_abbr)
    home_code = retro_code(home_abbr)

    away_runs = int(game["away_score"])
    home_runs = int(game["home_score"])

    if away_runs == home_runs:
        raise ValueError(f"Completed game {game['game_id']} has a tied final score.")

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
    """Load valid prior output; ignore an absent, empty, or corrupt parquet."""
    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_parquet(OUTPUT_PATH)
    except Exception as exc:
        print(f"Warning: ignoring unreadable existing output ({OUTPUT_PATH.name}): {exc}")
        return pd.DataFrame()


def main() -> None:
    today_et = datetime.datetime.now(ET).date()
    last_completed_date = min(today_et - datetime.timedelta(days=1), datetime.date(SEASON, 12, 31))
    season_start = datetime.date(SEASON, 3, 1)

    if last_completed_date < season_start:
        print(f"No completed {SEASON} regular-season games are available yet.")
        return

    print(
        f"Fetching completed MLB regular-season games: {season_start} through {last_completed_date}"
    )

    games = fetch_completed_games(season_start, last_completed_date)
    rows = [normalize_game(game) for game in games]

    if not rows:
        print("No completed games returned from MLB Stats API.")
        return

    fresh = pd.DataFrame(rows)

    team_columns = ["visteam", "hometeam", "wteam"]
    blank_team_mask = fresh[team_columns].replace("", pd.NA).isna().any(axis=1)

    if blank_team_mask.any():
        raise ValueError(
            "Refusing to write games with blank team codes:\n"
            + fresh.loc[blank_team_mask].to_string(index=False)
        )

    existing = load_existing()

    combined = pd.concat(
        [existing, fresh],
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined.sort_values(["game_id", "retrieved_at_utc"])
        .drop_duplicates(subset=["game_id"], keep="last")
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)

    coverage_dates = pd.to_datetime(
        combined["date"].astype(str),
        format="%Y%m%d",
    )

    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Games: {len(combined):,}")
    print(f"First game: {coverage_dates.min().date()}")
    print(f"Last game: {coverage_dates.max().date()}")


if __name__ == "__main__":
    main()
