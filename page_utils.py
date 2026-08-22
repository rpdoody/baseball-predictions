"""
Shared utilities, constants, and cached data loaders for all Streamlit pages.
Import from this module instead of duplicating code across pages.
"""

import datetime
import json
import urllib.parse
import urllib.request
import math
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent.resolve()
# ROOT first on sys.path so local src/ package is found before any PyPI 'src'
sys.path.insert(0, str(ROOT))

import statsapi

# Re-exported for pages and tests that import it from page_utils.
from footer import add_betting_oracle_footer  # noqa: F401
from retrosheet import TEAM_NAMES


PROCESSED = ROOT / "data_files" / "processed"
ET = ZoneInfo("America/New_York")


# ─── Brand Colors ─────────────────────────────────────────────────────────────
MLB_BLUE = "#002D72"
MLB_RED = "#D50032"
DARK_GREEN = "#1a4731"


CONF_COLORS: dict[str, str] = {
    "HIGH": "#16a34a",
    "MEDIUM": "#d97706",
    "LOW": "#6b7280",
    "High": "#16a34a",
    "Medium": "#d97706",
    "Low": "#6b7280",
}


# ─── MLB full-name → Retrosheet short-name ────────────────────────────────────
_MLB_TO_RETRO: dict[str, str] = {
    "Arizona Diamondbacks": "Diamondbacks",
    "Atlanta Braves": "Braves",
    "Baltimore Orioles": "Orioles",
    "Boston Red Sox": "Red Sox",
    "Chicago Cubs": "Cubs",
    "Chicago White Sox": "White Sox",
    "Cincinnati Reds": "Reds",
    "Cleveland Guardians": "Guardians",
    "Colorado Rockies": "Rockies",
    "Detroit Tigers": "Tigers",
    "Houston Astros": "Astros",
    "Kansas City Royals": "Royals",
    "Los Angeles Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers",
    "Miami Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers",
    "Minnesota Twins": "Twins",
    "New York Mets": "Mets",
    "New York Yankees": "Yankees",
    "Oakland Athletics": "Athletics",
    "Sacramento Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies",
    "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres",
    "Seattle Mariners": "Mariners",
    "San Francisco Giants": "Giants",
    "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays",
    "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays",
    "Washington Nationals": "Nationals",
}


# ─── Human-readable column headers ────────────────────────────────────────────
READABLE_COLS: dict[str, str] = {
    "team": "Team",
    "G": "Games",
    "W": "W",
    "L": "L",
    "WPct": "Win %",
    "PythWPct": "Pythagorean W%",
    "Home_W": "Home W",
    "Home_L": "Home L",
    "Away_W": "Away W",
    "Away_L": "Away L",
    "RS": "Runs Scored",
    "RA": "Runs Allowed",
    "RD": "Run Diff",
    "RD_per_G": "Run Diff / G",
    "RS_per_G": "RS / G",
    "RA_per_G": "RA / G",
    "PA": "Plate App",
    "AB": "At Bats",
    "R": "Runs",
    "H": "Hits",
    "HR": "HR",
    "RBI": "RBI",
    "BB": "Walks",
    "K": "K",
    "SB": "Stolen Bases",
    "BA": "AVG",
    "SLG": "SLG",
    "OPS": "OPS",
    "IP": "Inn. Pitched",
    "HA": "Hits Allowed",
    "HRA": "HR Allowed",
    "ER": "Earned Runs",
    "SO": "Strikeouts",
    "ERA": "ERA",
    "WHIP": "WHIP",
    "K9": "K / 9",
    "BB9": "BB / 9",
    "HR9": "HR / 9",
    "K_BB": "K / BB",
    "GS": "Starts",
    "full_name": "Player",
    "date": "Date",
    "visteam": "Visitor",
    "hometeam": "Home",
    "vruns": "Visitor Runs",
    "hruns": "Home Runs",
}


# ─── Cached API / Data Loaders ────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_todays_schedule(
    game_date: datetime.date | None = None,
) -> list[dict]:
    """Fetch the complete MLB schedule for an explicit Eastern game date."""
    try:
        target_date = game_date or datetime.datetime.now(ET).date()
        games = statsapi.schedule(
            date=target_date.isoformat(),
            sportId=1,
        ) or []
        allowed_game_types = {"R", "F", "D", "L", "W", "S"}
        return [
            game
            for game in games
            if game.get("game_type", "R") in allowed_game_types
        ]
    except Exception as exc:
        print(f"MLB schedule fetch failed for {game_date}: {exc}")
        return []

@st.cache_data(show_spinner=False, ttl=300)
def _fetch_confirmed_lineups(game_pk: int) -> dict:
    """Fetch official MLB batting orders when both lineups are complete."""
    empty = {"away": [], "home": [], "confirmed": False}

    if not game_pk:
        return empty

    try:
        data = statsapi.get(
            "game",
            {
                "gamePk": int(game_pk),
                "fields": (
                    "liveData,boxscore,teams,players,battingOrder,"
                    "person,fullName,position"
                ),
            },
        )
        teams = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
        lineups: dict[str, list[dict]] = {}

        for side in ("away", "home"):
            players = teams.get(side, {}).get("players", {})
            batting_order: list[dict] = []

            for player in players.values():
                order_raw = str(player.get("battingOrder") or "").strip()

                # MLB uses values such as "100", "200", ..., "900".
                if not order_raw or order_raw == "000":
                    continue

                try:
                    order = int(order_raw) // 100
                except ValueError:
                    continue

                person = player.get("person", {})
                position = player.get("position", {})
                batting_order.append(
                    {
                        "order": order,
                        "player": person.get("fullName", "Unknown"),
                        "position": position.get("abbreviation", "—"),
                    }
                )

            lineups[side] = sorted(batting_order, key=lambda row: row["order"])

        # Require nine unique lineup spots per side. This avoids displaying
        # incomplete projected/partial batting orders as confirmed.
        expected_slots = set(range(1, 10))
        is_confirmed = (
            expected_slots.issubset({row["order"] for row in lineups["away"]})
            and expected_slots.issubset({row["order"] for row in lineups["home"]})
        )

        if not is_confirmed:
            return empty

        return {
            "away": lineups["away"],
            "home": lineups["home"],
            "confirmed": True,
        }
    except Exception:
        return empty
    
@st.cache_data(show_spinner=False, ttl=86400)
def _load_latest_odds() -> pd.DataFrame:
    """Return odds from the latest morning-pipeline CSV without a live API call."""
    odds_dir = ROOT / "data_files" / "raw" / "odds"
    if not odds_dir.exists():
        return pd.DataFrame()
    files = sorted(odds_dir.glob("odds_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


# ─── Game Context Helpers ─────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=86400)
def _load_game_context_cache() -> dict:
    """Precompute Retrosheet-derived park, umpire, bullpen, and platoon context."""
    retro_dir = ROOT / "data_files" / "retrosheet"
    code_to_short = {code: short_name for code, short_name in TEAM_NAMES.items()}
    out: dict = {
        "park_factors": {},
        "daynight": {},
        "bullpen_ip_pg": {},
        "platoon": {},
    }

    try:
        gi_raw = pd.read_parquet(retro_dir / "gameinfo.parquet")
        gi_raw["vruns"] = pd.to_numeric(gi_raw["vruns"], errors="coerce")
        gi_raw["hruns"] = pd.to_numeric(gi_raw["hruns"], errors="coerce")
        gi_raw["total_runs"] = gi_raw["vruns"] + gi_raw["hruns"]
    except Exception:
        gi_raw = pd.DataFrame()

    try:
        gi = gi_raw.copy()
        max_season = int(gi["season"].max())
        recent = gi[gi["season"] >= max_season - 2].copy()
        league_rpg = recent["total_runs"].mean()
        park = recent.groupby("hometeam", observed=False).agg(games=("gid", "count"), runs=("total_runs", "sum")).reset_index()
        park = park[park["games"] >= 20]
        park["pf"] = (park["runs"] / park["games"] / league_rpg).round(3)
        park["short"] = park["hometeam"].map(code_to_short)
        out["park_factors"] = dict(zip(park["short"], park["pf"]))

        if "wteam" in gi.columns and "daynight" in gi.columns:
            daynight_rows = []
            for team_column in ("visteam", "hometeam"):
                temp = gi[["season", team_column, "daynight", "wteam"]].copy()
                temp.columns = ["season", "team", "dn", "wteam"]
                temp["won"] = (temp["wteam"] == temp["team"]).astype(int)
                daynight_rows.append(temp)
            daynight = pd.concat(daynight_rows, ignore_index=True)
            daynight["dn"] = daynight["dn"].fillna("n").str.lower().str.strip()
            daynight = daynight[daynight["season"] >= max_season - 2]
            grouped = daynight.groupby(["team", "dn"], observed=False).agg(games=("won", "count"), wins=("won", "sum")).reset_index()
            grouped["wpct"] = (grouped["wins"] / grouped["games"].clip(lower=1)).round(3)
            grouped["short"] = grouped["team"].map(code_to_short)
            for _, row in grouped.iterrows():
                short_name = row["short"]
                if short_name:
                    out["daynight"].setdefault(short_name, {})[row["dn"]] = row["wpct"]
    except Exception:
        pass

    try:
        csv_path = retro_dir / "gameinfo.csv"
        if csv_path.exists():
            gi_csv = pd.read_csv(csv_path, low_memory=False, usecols=lambda column: column in {"gid", "hometeam", "vruns", "hruns", "season"})
        else:
            needed = [column for column in ("gid", "hometeam", "vruns", "hruns", "season") if column in gi_raw.columns]
            gi_csv = gi_raw[needed].copy()
        gi_csv["season"] = pd.to_numeric(gi_csv["season"], errors="coerce")
        gi_csv["vruns"] = pd.to_numeric(gi_csv["vruns"], errors="coerce")
        gi_csv["hruns"] = pd.to_numeric(gi_csv["hruns"], errors="coerce")
        gi_csv["total_runs"] = gi_csv["vruns"] + gi_csv["hruns"]
        max_csv_season = int(gi_csv["season"].dropna().max())
        recent_csv = gi_csv[gi_csv["season"] >= max_csv_season - 2]
        park_runs = recent_csv.groupby("hometeam", observed=False)["total_runs"].agg(["mean", "count"]).reset_index()
        park_runs.columns = ["team", "avg_runs", "games"]
        park_runs = park_runs[park_runs["games"] >= 20]
        park_runs["short"] = park_runs["team"].map(code_to_short)
        out["ump_park_avg"] = {row["short"]: round(row["avg_runs"], 2) for _, row in park_runs.iterrows() if row.get("short")}
    except Exception:
        pass

    try:
        gi_ump = gi_raw.copy()
        if "umphome" in gi_ump.columns:
            gi_ump["date"] = pd.to_datetime(gi_ump["date"].astype(str), format="%Y%m%d", errors="coerce")
            gi_ump = gi_ump.sort_values("date").reset_index(drop=True)
            league_ump_mean = float(gi_ump["total_runs"].mean())
            ump_grouped = gi_ump.groupby("umphome", observed=False)["total_runs"].agg(runs_avg="mean", games="count").reset_index()
            ump_grouped.columns = ["ump_id", "runs_avg", "games"]
            ump_grouped["over_mean"] = (ump_grouped["runs_avg"] - league_ump_mean).round(2)
            ump_grouped["above_avg"] = ump_grouped["runs_avg"] > league_ump_mean

            def ump_trend(group: pd.DataFrame) -> float:
                values = group.sort_values("date")["total_runs"].values
                return round(float(values[-20:].mean() - values[-40:-20].mean()), 2) if len(values) >= 40 else 0.0

            trends = gi_ump.groupby("umphome", observed=False).apply(ump_trend, include_groups=False).reset_index()
            trends.columns = ["ump_id", "trend"]
            ump_grouped = ump_grouped.merge(trends, on="ump_id", how="left")
            ump_grouped["trend"] = ump_grouped["trend"].fillna(0.0)
            out["umpire_stats"] = {
                row["ump_id"]: {
                    "runs_avg": round(float(row["runs_avg"]), 2),
                    "over_mean": float(row["over_mean"]),
                    "games": int(row["games"]),
                    "above_avg": bool(row["above_avg"]),
                    "trend": float(row["trend"]),
                }
                for _, row in ump_grouped.iterrows()
                if pd.notna(row["ump_id"])
            }
    except Exception:
        pass

    try:
        pitching = pd.read_parquet(retro_dir / "pitching.parquet")
        pitching = pitching[pitching["p_gs"] != 1.0].copy()
        pitching["season"] = pd.to_numeric(pitching["date"].astype(str).str[:4], errors="coerce")
        max_pitching_season = int(pitching["season"].dropna().max())
        pitching = pitching[pitching["season"] >= max_pitching_season - 1]
        pitching["ip"] = pd.to_numeric(pitching["p_ipouts"], errors="coerce").fillna(0) / 3
        bullpen = pitching.groupby("team", observed=False).agg(total_ip=("ip", "sum"), total_games=("gid", "nunique")).reset_index()
        bullpen["ip_pg"] = (bullpen["total_ip"] / bullpen["total_games"].clip(lower=1)).round(2)
        bullpen["short"] = bullpen["team"].map(code_to_short)
        out["bullpen_ip_pg"] = {row["short"]: row["ip_pg"] for _, row in bullpen.iterrows() if row.get("short")}
    except Exception:
        pass

    try:
        players = pd.read_parquet(retro_dir / "allplayers.parquet")
        if {"season", "bat", "team"}.issubset(players.columns):
            max_player_season = int(players["season"].dropna().max())
            players = players[players["season"] >= max_player_season - 1]
            platoon = players.groupby("team", observed=False).apply(lambda group: pd.Series({"pct_left": round((group["bat"] == "L").mean(), 3), "pct_right": round((group["bat"] == "R").mean(), 3)}), include_groups=False).reset_index()
            platoon["short"] = platoon["team"].map(code_to_short)
            out["platoon"] = {row["short"]: {"pct_left": row["pct_left"], "pct_right": row["pct_right"]} for _, row in platoon.iterrows() if row.get("short")}
    except Exception:
        pass

    return out


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_game_umpires(game_pk: int) -> dict:
    """Fetch the umpire crew for a specific game from the MLB Stats API."""
    if not game_pk:
        return {}
    try:
        data = statsapi.get("game", {"gamePk": game_pk, "fields": "liveData,boxscore,officials"})
        officials = data.get("liveData", {}).get("boxscore", {}).get("officials", [])
        type_map = {"Home Plate": "home_plate", "First Base": "first", "Second Base": "second", "Third Base": "third"}
        result: dict = {}
        for official in officials:
            official_type = official.get("officialType", "")
            name = official.get("official", {}).get("fullName", "")
            if official_type in type_map and name:
                result[type_map[official_type]] = name
        return result
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_retrosheet_game_umpires(home_retro: str, away_retro: str, game_date: str) -> dict:
    """Locate historic game umpire IDs from Retrosheet gameinfo.parquet."""
    try:
        if not home_retro or not away_retro or not game_date:
            return {}
        retro_dir = ROOT / "data_files" / "retrosheet"
        gameinfo = pd.read_parquet(retro_dir / "gameinfo.parquet")
        if not {"date", "hometeam", "visteam"}.issubset(gameinfo.columns):
            return {}
        code_to_short = {code: short_name for code, short_name in TEAM_NAMES.items()}
        short_to_code = {short_name: code for code, short_name in code_to_short.items()}
        home_code = short_to_code.get(home_retro)
        away_code = short_to_code.get(away_retro)
        if not home_code or not away_code:
            return {}
        gameinfo = gameinfo.copy()
        gameinfo["game_date"] = pd.to_datetime(gameinfo["date"].astype(str), format="%Y%m%d", errors="coerce").dt.date
        target_date = pd.to_datetime(game_date, errors="coerce").date()
        if pd.isna(target_date):
            return {}
        match = gameinfo[(gameinfo["hometeam"] == home_code) & (gameinfo["visteam"] == away_code) & (gameinfo["game_date"] == target_date)]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {
            "home_plate": str(row.get("umphome", "")).strip() if not pd.isna(row.get("umphome", "")) else "",
            "first": str(row.get("ump1b", "")).strip() if not pd.isna(row.get("ump1b", "")) else "",
            "second": str(row.get("ump2b", "")).strip() if not pd.isna(row.get("ump2b", "")) else "",
            "third": str(row.get("ump3b", "")).strip() if not pd.isna(row.get("ump3b", "")) else "",
        }
    except Exception:
        return {}


def _lookup_ump_retro_id(name: str, ump_stats: dict) -> str | None:
    """Fuzzy-match an umpire full name or Retrosheet ID to cached stats."""
    if not name or not ump_stats:
        return None
    if name in ump_stats:
        return name
    parts = name.lower().split()
    if len(parts) < 2:
        return None
    last4 = parts[-1][:4]
    first1 = parts[0][0]
    prefix = last4 + first1
    for umpire_id in ump_stats:
        if umpire_id.lower().startswith(prefix):
            return umpire_id
    for umpire_id in ump_stats:
        if umpire_id.lower().startswith(last4):
            return umpire_id
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_pitcher_throw_hand(pitcher_name: str) -> str:
    """Return pitcher throwing hand: L, R, or ? via MLB Stats API."""
    if not pitcher_name or pitcher_name.strip().upper() == "TBD":
        return "?"
    try:
        results = statsapi.lookup_player(pitcher_name)
        if not results:
            return "?"
        data = statsapi.get("people", {"personIds": results[0]["id"]})
        if data and data.get("people"):
            return data["people"][0].get("pitchHand", {}).get("code", "?")
    except Exception:
        pass
    return "?"


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_team_il_players(team_full_name: str) -> list[str]:
    """Return injured-list player names for a team from MLB Stats API."""
    try:
        results = statsapi.lookup_team(team_full_name)
        if not results:
            return []
        roster = statsapi.get("roster", {"teamId": results[0]["id"], "rosterType": "injuredList"})
        return [player["person"]["fullName"] for player in roster.get("roster", [])]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_team_rest_days(team_full_name: str) -> int | None:
    """Return calendar days since the team's most recent completed game."""
    try:
        results = statsapi.lookup_team(team_full_name)
        if not results:
            return None
        team_id = results[0]["id"]
        today = datetime.datetime.now(ET).date()
        start = (today - datetime.timedelta(days=10)).strftime("%m/%d/%Y")
        end = (today - datetime.timedelta(days=1)).strftime("%m/%d/%Y")
        schedule = statsapi.schedule(teamId=team_id, startDate=start, endDate=end) or []
        played = sorted(datetime.date.fromisoformat(game["game_date"]) for game in schedule if game.get("game_date") and game.get("status") not in {"Postponed", "Cancelled", "Suspended"})
        return (today - played[-1]).days if played else None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_team_standings(season: int | None = None) -> dict[str, dict]:
    """Return current-season MLB standings, with prior-season pregame fallback."""
    def parse(raw_standings) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for division_data in raw_standings.values():
            for team in division_data.get("teams", []):
                wins = team.get("w", 0) or 0
                losses = team.get("l", 0) or 0
                pct = team.get("pct")
                if pct in (None, "—", ""):
                    pct = round(wins / (wins + losses), 3) if wins + losses else 0.500
                result[team["name"]] = {"W": wins, "L": losses, "pct": pct, "streak": team.get("streak", "—"), "L10": team.get("lastTen", "—")}
        return result

    try:
        current_season = season or datetime.datetime.now(ET).year
        result = parse(statsapi.standings_data(season=current_season))
        total_wins = sum(int(value["W"]) for value in result.values() if str(value.get("W", "")).isdigit())
        if total_wins == 0 and season is None:
            prior = parse(statsapi.standings_data(season=current_season - 1))
            if prior:
                return prior
        return result
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_pitcher_stats(pitcher_name: str) -> dict:
    """Return season pitching statistics for a named pitcher from MLB Stats API."""
    if not pitcher_name or pitcher_name.strip().upper() == "TBD":
        return {}
    try:
        results = statsapi.lookup_player(pitcher_name)
        if not results:
            return {}
        data = statsapi.player_stat_data(results[0]["id"], group="pitching", type="season", sportId=1)
        if not data or not data.get("stats"):
            return {}
        stats = data["stats"][0]["stats"]
        return {
            "W-L": f"{stats.get('wins', '?')}-{stats.get('losses', '?')}",
            "ERA": str(stats.get("era", "—")),
            "IP": str(stats.get("inningsPitched", "—")),
            "GS": str(stats.get("gamesStarted", "—")),
            "K": str(stats.get("strikeOuts", "—")),
            "BB": str(stats.get("baseOnBalls", "—")),
            "HR": str(stats.get("homeRuns", "—")),
            "WHIP": str(stats.get("whip", "—")),
            "K/9": str(stats.get("strikeoutsPer9Inn", "—")),
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_espn_odds(
    game_date: datetime.date | None = None,
) -> list[dict]:
    """Fetch ESPN MLB odds for an explicit Eastern-date slate.

    Defaults to today's ET date when no date is supplied, preserving the
    previous zero-argument call signature used elsewhere in the app.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests

    target_date = game_date or datetime.datetime.now(ET).date()
    date_str = target_date.strftime("%Y%m%d")

    try:
        response = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
            f"?dates={date_str}&limit=30",
            timeout=10,
        )
        response.raise_for_status()
        events = response.json().get("events", [])
    except Exception as exc:
        print(f"ESPN odds scoreboard fetch failed for {target_date}: {exc}")
        return []

    def fetch_one(event: dict) -> dict | None:
        try:
            competition = event["competitions"][0]
            event_id = event["id"]
            competition_id = competition["id"]
            home_name = next((competitor["team"]["displayName"] for competitor in competition["competitors"] if competitor["homeAway"] == "home"), "")
            away_name = next((competitor["team"]["displayName"] for competitor in competition["competitors"] if competitor["homeAway"] == "away"), "")
            odds_response = requests.get(
                "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
                f"/events/{event_id}/competitions/{competition_id}/odds",
                timeout=10,
            )
            odds_response.raise_for_status()
            items = odds_response.json().get("items", [])
            if not items:
                return None
            odds = items[0]
            home_odds = odds.get("homeTeamOdds", {})
            away_odds = odds.get("awayTeamOdds", {})
            return {
                "home_team": home_name,
                "away_team": away_name,
                "provider": odds.get("provider", {}).get("name", "ESPN"),
                "ml_home": home_odds.get("moneyLine"),
                "ml_away": away_odds.get("moneyLine"),
                "spread_home": home_odds.get("current", {}).get("spread", {}).get("american", "—"),
                "spread_away": away_odds.get("current", {}).get("spread", {}).get("american", "—"),
                "details": odds.get("details", "—"),
                "over_under": odds.get("overUnder"),
                "over_odds": odds.get("overOdds", "—"),
                "under_odds": odds.get("underOdds", "—"),
            }
        except Exception:
            return None

    result = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one, event) for event in events]
        for future in as_completed(futures):
            item = future.result()
            if item:
                result.append(item)
    return result

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_action_public_betting(
    game_date: datetime.date | None = None,
) -> list[dict]:
    """
    Retrieve MLB public betting splits from Action Network.

    The data is provider-specific public betting data. Missing/locked fields
    are returned as None rather than guessed.
    """
    target_date = game_date or datetime.datetime.now(ET).date()
    params = {
        "period": "game",
        "date": target_date.strftime("%Y%m%d"),
    }
    url = (
        "https://api.actionnetwork.com/web/v1/scoreboard/publicbetting/mlb?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://www.actionnetwork.com",
            "Referer": "https://www.actionnetwork.com/mlb/public-betting",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    games = payload.get("games", payload.get("events", []))
    rows: list[dict] = []

    def _num(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _market_side(market: dict, side: str) -> dict:
        side_data = market.get(side, {}) if isinstance(market, dict) else {}
        return {
            "bets_pct": _num(
                side_data.get("bets_percentage", side_data.get("bet_percentage"))
            ),
            "money_pct": _num(
                side_data.get("money_percentage", side_data.get("money_percent"))
            ),
        }

    for event in games:
        teams = event.get("teams", [])
        home = next(
            (
                team for team in teams
                if str(team.get("side", "")).lower() == "home"
            ),
            {},
        )
        away = next(
            (
                team for team in teams
                if str(team.get("side", "")).lower() == "away"
            ),
            {},
        )

        # Action response structures can vary; inspect raw output if this
        # returns no usable markets for a date.
        markets = event.get("markets", event.get("odds", {}))
        moneyline = (
            markets.get("moneyline")
            or markets.get("money_line")
            or event.get("moneyline")
            or {}
        )

        home_splits = _market_side(moneyline, "home")
        away_splits = _market_side(moneyline, "away")

        rows.append(
            {
                "home_team": home.get("full_name") or home.get("name"),
                "away_team": away.get("full_name") or away.get("name"),
                "event_id": event.get("id"),
                "home_ml_bets_pct": home_splits["bets_pct"],
                "home_ml_money_pct": home_splits["money_pct"],
                "away_ml_bets_pct": away_splits["bets_pct"],
                "away_ml_money_pct": away_splits["money_pct"],
            }
        )

    return rows

@st.cache_data(show_spinner=False)
def _load_precomputed() -> dict:
    """Load all precomputed Retrosheet and model datasets once per process."""
    gameinfo = pd.read_parquet(ROOT / "data_files" / "retrosheet" / "gameinfo.parquet")
    mc_ranking = pd.read_csv(PROCESSED / "mc_feature_ranking.csv") if (PROCESSED / "mc_feature_ranking.csv").exists() else None
    mc_trials = pd.read_parquet(PROCESSED / "mc_feature_trials.parquet") if (PROCESSED / "mc_feature_trials.parquet").exists() else None
    savant_metrics = pd.read_parquet(PROCESSED / "savant_model_metrics.parquet") if (PROCESSED / "savant_model_metrics.parquet").exists() else None
    savant_imps = pd.read_parquet(PROCESSED / "savant_model_importances.parquet") if (PROCESSED / "savant_model_importances.parquet").exists() else None
    return {
        "gameinfo": gameinfo,
        "standings": pd.read_parquet(PROCESSED / "standings.parquet"),
        "team_batting": pd.read_parquet(PROCESSED / "team_batting.parquet"),
        "team_pitching": pd.read_parquet(PROCESSED / "team_pitching.parquet"),
        "batting_leaders": pd.read_parquet(PROCESSED / "batting_leaders.parquet"),
        "pitching_leaders": pd.read_parquet(PROCESSED / "pitching_leaders.parquet"),
        "model_features": pd.read_parquet(PROCESSED / "model_features.parquet"),
        "mc_ranking": mc_ranking,
        "mc_trials": mc_trials,
        "savant_metrics": savant_metrics,
        "savant_imps": savant_imps,
    }


@st.cache_data(show_spinner=False)
def _load_model_results() -> dict | None:
    """Load precomputed model evaluation outputs."""
    try:
        metrics_df = pd.read_parquet(PROCESSED / "model_metrics.parquet")
        importances_df = pd.read_parquet(PROCESSED / "model_importances.parquet")
        results = {}
        for model_name in ["moneyline", "spread", "totals"]:
            row = metrics_df[metrics_df["model"] == model_name].iloc[0]
            results[model_name] = {
                "model": None,
                "metrics": {
                    "roc_auc": float(row["roc_auc"]),
                    "accuracy": float(row["accuracy"]),
                    "brier_score": float(row["brier_score"]),
                    "log_loss": float(row["log_loss"]),
                },
                "importances": importances_df[importances_df["model"] == model_name][["feature", "importance"]].reset_index(drop=True),
                "feature_cols": [],
                "test_df": pd.read_parquet(PROCESSED / f"{model_name}_test_df.parquet"),
                "train_size": int(row["train_size"]),
                "test_size": int(row["test_size"]),
            }
        return results
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def _load_eval_backtests() -> dict | None:
    """Load precomputed backtest objects."""
    try:
        from src.evaluation.backtester import BacktestResult, BetResult
        bets_df = pd.read_parquet(PROCESSED / "backtest_bets.parquet")
        summary_df = pd.read_parquet(PROCESSED / "backtest_summary.parquet")
        backtests = {}
        for model_name in ["moneyline", "totals"]:
            subset = bets_df[bets_df["model_name"] == model_name]
            summary = summary_df[summary_df["model"] == model_name].iloc[0]
            bets = [BetResult(game_id=row.game_id, date=row.date, pick_type=row.pick_type, pick_value="", predicted_prob=float(row.predicted_prob), confidence_score=float(row.confidence_score), confidence=row.confidence, edge=float(row.edge), american_odds=int(row.american_odds), result=row.result, profit_units=float(row.profit_units)) for row in subset.itertuples(index=False)]
            backtests[model_name] = BacktestResult(model_name=model_name, pick_type=str(summary["pick_type"]), period=str(summary["period"]), bets=bets)
        return backtests
    except FileNotFoundError:
        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _kelly_fraction(prob: float, american_odds: int) -> float:
    """Return full-Kelly wager fraction."""
    decimal = american_odds / 100.0 + 1.0 if american_odds >= 0 else 100.0 / abs(american_odds) + 1.0
    b = decimal - 1.0
    return max((b * prob - (1.0 - prob)) / b, 0.0)


def get_dataframe_height(df: pd.DataFrame, row_height: int = 35, header_height: int = 38, padding: int = 2, max_height: int = 600) -> int:
    """Calculate optimal Streamlit dataframe height in pixels."""
    calculated = len(df) * row_height + header_height + padding
    return min(calculated, max_height) if max_height else calculated


def _american_to_implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability without de-vigging."""
    return 100.0 / (american_odds + 100.0) if american_odds >= 0 else abs(american_odds) / (abs(american_odds) + 100.0)


def _estimate_win_prob(home_full: str, away_full: str, live_standings: dict[str, dict]) -> float:
    """Quick current-record logistic estimate with home-field adjustment."""
    def pct(name: str) -> float:
        data = live_standings.get(name, {})
        try:
            value = data.get("pct", 0.500)
            return float(value) if float(value) > 0 else 0.500
        except (TypeError, ValueError):
            return 0.500
    difference = pct(home_full) - pct(away_full) + 0.04
    return max(0.10, min(0.90, 1.0 / (1.0 + math.exp(-difference * 8))))


# ─── HTML Components ──────────────────────────────────────────────────────────


def _prob_bar_html(home_prob: float, home: str, away: str) -> str:
    """Inline HTML home/away win-probability bar."""
    home_pct = round(home_prob * 100)
    away_pct = 100 - home_pct
    return (
        f'<div style="display:flex;height:22px;border-radius:6px;overflow:hidden;font-size:0.75rem;font-weight:600">'
        f'<div style="width:{home_pct}%;background:{MLB_BLUE};color:white;display:flex;align-items:center;justify-content:center">{home_pct}%</div>'
        f'<div style="width:{away_pct}%;background:{MLB_RED};color:white;display:flex;align-items:center;justify-content:center">{away_pct}%</div>'
        f'</div><div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#888;margin-top:2px"><span>{home} (home)</span><span>{away} (away)</span></div>'
    )


def _conf_badge(tier: str) -> str:
    color = CONF_COLORS.get(tier, "#6b7280")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:10px;font-size:0.72rem;font-weight:700">{tier.upper()}</span>'


# ─── Sidebar ──────────────────────────────────────────────────────────────────


def render_sidebar(show_year_filter: bool = True) -> tuple[int, int]:
    """Render branding and an optional Eastern-date year-range filter."""
    with st.sidebar:
        logo = ROOT / "data_files" / "logo.png"
        if logo.exists():
            _, center, _ = st.columns([1, 2, 1])
            with center:
                st.image(str(logo), width=150)
        if show_year_filter:
            st.markdown("---")
            st.header("Season Filters")
            current_year = datetime.datetime.now(ET).year
            min_year, max_year = st.slider("Season range", min_value=2020, max_value=current_year, value=(2020, current_year), step=1)
            st.caption(f"Using {min_year}–{max_year} regular-season games.")
            return min_year, max_year
    current_year = datetime.datetime.now(ET).year
    return 2020, current_year


# ─── Session State ────────────────────────────────────────────────────────────


def init_session_state(features_df: pd.DataFrame | None = None) -> None:
    """Prepopulate session state from precomputed results on first load."""
    if "ml_results" not in st.session_state:
        st.session_state["ml_results"] = _load_model_results()
    if features_df is not None and "ml_feat_df" not in st.session_state:
        st.session_state["ml_feat_df"] = features_df
    if "eval_backtests" not in st.session_state:
        st.session_state["eval_backtests"] = _load_eval_backtests()
    if "schedule_selected_game" not in st.session_state:
        st.session_state["schedule_selected_game"] = None