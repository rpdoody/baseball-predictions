from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any

import pandas as pd

from src.models.score_distribution import (
    ScoreDistribution,
    independent_poisson_score_distribution,
)

LEAGUE_RPG = 4.5
HOME_FIELD_RUNS = 0.14


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def last_word(name: str) -> str:
    return name.split()[-1] if name else ""


@dataclass
class ContextualProjection:
    away_runs: float
    home_runs: float
    away_win_probability: float
    home_win_probability: float
    total_runs: float
    distribution: ScoreDistribution
    adjustments: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def team_historical_row(team_full: str, hist_stnd: pd.DataFrame) -> pd.Series | None:
    """Find most recent historical standings row for an MLB full team name."""
    team_token = last_word(team_full)

    try:
        subset = hist_stnd[
            hist_stnd["team"].str.contains(
                team_token,
                case=False,
                na=False,
            )
        ]
        if subset.empty:
            return None

        return subset.sort_values("season").iloc[-1]
    except (KeyError, AttributeError):
        return None


def baseline_runs(
    offense_team: str,
    defense_team: str,
    hist_stnd: pd.DataFrame,
) -> float:
    """
    Baseline expected runs from most recent RS/G and opponent RA/G.

    Uses geometric blending followed by shrinkage to league average.
    """
    offense_row = team_historical_row(offense_team, hist_stnd)
    defense_row = team_historical_row(defense_team, hist_stnd)

    if offense_row is None or defense_row is None:
        return LEAGUE_RPG

    rs_g = safe_float(offense_row.get("RS_per_G"), LEAGUE_RPG)
    ra_g = safe_float(defense_row.get("RA_per_G"), LEAGUE_RPG)

    raw_rate = (max(rs_g, 0.1) * max(ra_g, 0.1)) ** 0.5
    return 0.70 * raw_rate + 0.30 * LEAGUE_RPG


def pitcher_run_multiplier(pitcher_stats: dict | None) -> tuple[float, str | None]:
    """
    Conservative starter adjustment based on currently available Stats API data.

    Below 1.0 reduces opponent projected scoring. Above 1.0 increases it.
    Uses ERA and WHIP only because those are the validated fields currently
    returned by _fetch_pitcher_stats().
    """
    if not pitcher_stats:
        return 1.0, "Starter data unavailable; neutral starter adjustment applied."

    era = safe_float(pitcher_stats.get("ERA"), LEAGUE_RPG)
    whip = safe_float(pitcher_stats.get("WHIP"), 1.30)

    era_factor = clamp(era / 4.20, 0.82, 1.22)
    whip_factor = clamp(whip / 1.30, 0.88, 1.15)

    return clamp(0.70 * era_factor + 0.30 * whip_factor, 0.82, 1.20), None


def bullpen_run_multiplier(
    opponent_retro: str,
    game_context: dict,
) -> tuple[float, str | None]:
    """
    Use bullpen IP per game as a small workload proxy.

    This is not individual reliever availability. Treat it as a low-weight
    adjustment until last-1/3/7-day reliever workload is ingested.
    """
    ip_per_game = game_context.get("bullpen_ip_pg", {}).get(opponent_retro)

    if ip_per_game is None:
        return 1.0, "Bullpen workload unavailable; neutral bullpen adjustment applied."

    value = safe_float(ip_per_game, 3.1)
    multiplier = 1.0 + clamp((value - 3.1) * 0.025, -0.05, 0.05)

    return multiplier, "Bullpen adjustment uses historical relief IP/G, not live reliever availability."


def park_run_multiplier(home_retro: str, game_context: dict) -> tuple[float, str | None]:
    park_factor = game_context.get("park_factors", {}).get(home_retro)

    if park_factor is None:
        return 1.0, "Park factor unavailable; neutral park adjustment applied."

    return clamp(safe_float(park_factor, 1.0), 0.85, 1.15), None


def day_night_multiplier(
    team_retro: str,
    is_day_game: bool,
    game_context: dict,
) -> tuple[float, str | None]:
    """
    Convert historical day/night W% into a deliberately small run adjustment.

    Win percentage is an imperfect offensive proxy, so its effect is tightly
    capped and should be upgraded once team split offense is ingested.
    """
    split_key = "day" if is_day_game else "night"
    value = game_context.get("daynight", {}).get(team_retro, {}).get(split_key)

    if value is None:
        return 1.0, "Day/night split unavailable; neutral adjustment applied."

    win_pct = safe_float(value, 0.500)
    return 1.0 + clamp((win_pct - 0.500) * 0.16, -0.03, 0.03), None


def weather_run_multiplier(weather: dict | None) -> tuple[float, str | None]:
    """
    Conservative weather adjustment using the existing forecast schema.

    Assumed fields: is_dome, temp_f, humidity_pct, cloud_cover_pct.
    Wind direction remains intentionally neutral until stadium orientation and
    directional wind are added.
    """
    if not weather:
        return 1.0, "Weather unavailable; neutral weather adjustment applied."

    if weather.get("is_dome"):
        return 1.0, None

    temp_f = safe_float(weather.get("temp_f"), 70.0)
    humidity = safe_float(weather.get("humidity_pct"), 50.0)

    adjustment = 0.0
    adjustment += clamp((temp_f - 70.0) * 0.0025, -0.06, 0.06)
    adjustment += clamp((humidity - 50.0) * 0.0005, -0.015, 0.015)

    return 1.0 + adjustment, "Weather adjustment excludes wind direction."


def home_field_multiplier() -> float:
    return exp(HOME_FIELD_RUNS / LEAGUE_RPG)


def infer_day_game(game: dict) -> bool:
    """
    Infer game window from the scheduled UTC timestamp.

    The caller should pass an ET-normalized game hour when available. This
    approximation is acceptable only until game datetime parsing is centralized.
    """
    raw = game.get("game_datetime", "")
    try:
        dt = pd.Timestamp(raw)
        if dt.tzinfo is not None:
            dt = dt.tz_convert("America/New_York")
        return dt.hour < 17
    except (TypeError, ValueError):
        return False


def project_contextual_game(
    game: dict,
    hist_stnd: pd.DataFrame,
    game_context: dict,
    away_retro: str,
    home_retro: str,
    away_pitcher_stats: dict | None,
    home_pitcher_stats: dict | None,
    weather: dict | None,
) -> ContextualProjection:
    """
    Return coherent matchup probabilities from expected runs adjusted for
    park, weather, starter quality, bullpen proxy, day/night, and home field.
    """
    away_full = game.get("away_name", "")
    home_full = game.get("home_name", "")
    is_day_game = infer_day_game(game)

    away_base = baseline_runs(away_full, home_full, hist_stnd)
    home_base = baseline_runs(home_full, away_full, hist_stnd)

    home_starter_mult, home_starter_warning = pitcher_run_multiplier(home_pitcher_stats)
    away_starter_mult, away_starter_warning = pitcher_run_multiplier(away_pitcher_stats)

    home_bullpen_mult, home_bullpen_warning = bullpen_run_multiplier(
        home_retro,
        game_context,
    )
    away_bullpen_mult, away_bullpen_warning = bullpen_run_multiplier(
        away_retro,
        game_context,
    )

    park_mult, park_warning = park_run_multiplier(home_retro, game_context)
    weather_mult, weather_warning = weather_run_multiplier(weather)

    away_daynight_mult, away_daynight_warning = day_night_multiplier(
        away_retro,
        is_day_game,
        game_context,
    )
    home_daynight_mult, home_daynight_warning = day_night_multiplier(
        home_retro,
        is_day_game,
        game_context,
    )

    away_runs = away_base * home_starter_mult * home_bullpen_mult
    away_runs *= park_mult * weather_mult * away_daynight_mult

    home_runs = home_base * away_starter_mult * away_bullpen_mult
    home_runs *= park_mult * weather_mult * home_daynight_mult
    home_runs *= home_field_multiplier()

    away_runs = clamp(away_runs, 2.0, 8.0)
    home_runs = clamp(home_runs, 2.0, 8.0)

    distribution = independent_poisson_score_distribution(away_runs, home_runs)
    home_win = distribution.home_moneyline()
    away_win = 1.0 - home_win - distribution.tie_probability()

    warnings = [
        warning
        for warning in [
            home_starter_warning,
            away_starter_warning,
            home_bullpen_warning,
            away_bullpen_warning,
            park_warning,
            weather_warning,
            away_daynight_warning,
            home_daynight_warning,
        ]
        if warning
    ]

    return ContextualProjection(
        away_runs=away_runs,
        home_runs=home_runs,
        away_win_probability=away_win,
        home_win_probability=home_win,
        total_runs=away_runs + home_runs,
        distribution=distribution,
        adjustments={
            "away_baseline_runs": away_base,
            "home_baseline_runs": home_base,
            "park_multiplier": park_mult,
            "weather_multiplier": weather_mult,
            "home_field_multiplier": home_field_multiplier(),
            "away_starter_multiplier": away_starter_mult,
            "home_starter_multiplier": home_starter_mult,
            "away_bullpen_multiplier": away_bullpen_mult,
            "home_bullpen_multiplier": home_bullpen_mult,
            "away_daynight_multiplier": away_daynight_mult,
            "home_daynight_multiplier": home_daynight_mult,
        },
        warnings=warnings,
    )