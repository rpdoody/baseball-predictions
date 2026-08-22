"""Timestamped MLB lineup and probable-starter observations.

The module archives raw schedule and game-boxscore payloads before it normalizes
provider-neutral snapshots. A lineup is called confirmed only when the MLB game
boxscore supplies at least nine distinct batting-order slots. Each observation
has retrieval metadata so downstream point-in-time features can be reproduced.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import statsapi

from src.contracts.domain import stable_id
from src.ingestion.base import RetrievedPayload
from src.ingestion.raw_store import RawStore

from .config import config


MLB_LINEUP_SOURCE = "mlb_stats_lineups"
MLB_STARTER_SOURCE = "mlb_stats_starters"


def _archive_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    observed_at: datetime,
) -> Path | None:
    """Write an immutable, content-addressed normalized observation frame."""
    if frame.empty:
        return None

    content_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()
    target = (
        config.project_root
        / "data"
        / "silver"
        / source
        / f"observed_date={observed_at.date().isoformat()}"
        / f"{source}_{content_hash[:16]}.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    return target


def _require_aware(observed_at: datetime) -> None:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")


def _flatten_schedule_games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [game for date_group in payload.get("dates", []) for game in date_group.get("games", [])]


def _persist_raw_payload(
    *,
    source: str,
    body_object: Any,
    observed_at: datetime,
    request_params: dict[str, Any],
    run_id: str,
) -> str:
    """Archive raw provider content and return its content hash."""
    body = json.dumps(
        body_object,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    observation = RawStore(config.project_root / "data" / "bronze").persist(
        RetrievedPayload(
            source=source,
            body=body,
            observed_at=observed_at,
            request_params=request_params,
            http_metadata={"client": "MLB-StatsAPI"},
        ),
        ingestion_run_id=run_id,
    )
    return observation.payload_sha256


def _as_player_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _position_abbreviation(player: dict[str, Any]) -> str:
    position = player.get("position") or player.get("primaryPosition") or {}
    return str(position.get("abbreviation") or position.get("code") or "?")


def _bat_side(player: dict[str, Any]) -> str:
    return str((player.get("batSide") or {}).get("code") or "?").upper()


def _extract_batting_order(player: dict[str, Any]) -> int | None:
    """Return MLB batting-order slot from values such as 100, 200, ..., 900."""
    raw = player.get("battingOrder")
    if raw in (None, "", "0", 0):
        return None
    try:
        numeric = int(raw)
        order = numeric // 100 if numeric >= 100 else numeric
        return order if 1 <= order <= 9 else None
    except (TypeError, ValueError):
        return None


def _boxscore_players(
    boxscore: dict[str, Any],
    side: str,
) -> list[dict[str, Any]]:
    """Return the official batting-order player dictionaries for one team."""
    team = boxscore.get("teams", {}).get(side, {})
    players_by_key = team.get("players", {}) or {}
    batting_order_ids = team.get("battingOrder", []) or []

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Game boxscore battingOrder provides the authoritative current lineup order.
    for slot, raw_player_id in enumerate(batting_order_ids, start=1):
        player_id = _as_player_id(raw_player_id)
        player = players_by_key.get(f"ID{player_id}") or players_by_key.get(player_id) or {}
        person = player.get("person", {})
        resolved_id = _as_player_id(person.get("id") or raw_player_id)
        if not resolved_id or resolved_id in seen_ids:
            continue
        seen_ids.add(resolved_id)
        selected.append(
            {
                **player,
                "_resolved_player_id": resolved_id,
                "_resolved_batting_order": slot,
            }
        )

    # Fallback for feeds where battingOrder is absent but individual player
    # records include battingOrder values.
    if not selected:
        fallback = []
        for key, player in players_by_key.items():
            person = player.get("person", {})
            player_id = _as_player_id(person.get("id") or key.removeprefix("ID"))
            order = _extract_batting_order(player)
            if player_id and order is not None:
                fallback.append(
                    {
                        **player,
                        "_resolved_player_id": player_id,
                        "_resolved_batting_order": order,
                    }
                )
        selected = sorted(fallback, key=lambda player: player["_resolved_batting_order"])

    return selected


def _lineup_status(players: list[dict[str, Any]]) -> str:
    """Classify lineup completeness from distinct official batting-order slots."""
    slots = {
        player.get("_resolved_batting_order")
        for player in players
        if player.get("_resolved_batting_order") is not None
    }
    if set(range(1, 10)).issubset(slots):
        return "confirmed"
    if slots:
        return "partial"
    return "unavailable"


def _starter_row(
    *,
    game: dict[str, Any],
    side: str,
    observed_at: datetime,
    raw_payload_hash: str,
    ingestion_run_id: str,
) -> dict[str, object]:
    team = game.get("teams", {}).get(side, {})
    team_info = team.get("team", {}) or {}
    probable = team.get("probablePitcher", {}) or {}
    pitcher_id = _as_player_id(probable.get("id"))
    return {
        "game_id": int(game["gamePk"]),
        "game_datetime": game.get("gameDate", ""),
        "game_status": (game.get("status") or {}).get("detailedState", ""),
        "side": side,
        "team_id": _as_player_id(team_info.get("id")),
        "team_name": team_info.get("name", ""),
        "pitcher_id": pitcher_id,
        "pitcher_name": probable.get("fullName", ""),
        "starter_status": "probable" if pitcher_id else "unavailable",
        "observed_at": observed_at.isoformat(),
        "raw_payload_hash": raw_payload_hash,
        "ingestion_run_id": ingestion_run_id,
    }


def _lineup_rows_from_boxscore(
    *,
    game: dict[str, Any],
    boxscore: dict[str, Any],
    observed_at: datetime,
    raw_payload_hash: str,
    ingestion_run_id: str,
) -> list[dict[str, object]]:
    """Normalize one game's official boxscore batting orders into player rows."""
    rows: list[dict[str, object]] = []
    game_id = int(game["gamePk"])
    venue = game.get("venue", {}) or {}
    game_status = (game.get("status") or {}).get("detailedState", "")

    for side in ("away", "home"):
        schedule_team = game.get("teams", {}).get(side, {})
        team_info = schedule_team.get("team", {}) or {}
        players = _boxscore_players(boxscore, side)
        status = _lineup_status(players)

        for player in players:
            person = player.get("person", {}) or {}
            batting_order = player.get("_resolved_batting_order")
            if batting_order is None:
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "game_datetime": game.get("gameDate", ""),
                    "game_status": game_status,
                    "venue_id": _as_player_id(venue.get("id")),
                    "venue_name": venue.get("name", ""),
                    "side": side,
                    "team_id": _as_player_id(team_info.get("id")),
                    "team_name": team_info.get("name", ""),
                    "player_id": player.get("_resolved_player_id", ""),
                    "player_name": person.get("fullName", ""),
                    "bat_side": _bat_side(player),
                    "batting_order": int(batting_order),
                    "defensive_position": _position_abbreviation(player),
                    "lineup_status": status,
                    "observed_at": observed_at.isoformat(),
                    "raw_payload_hash": raw_payload_hash,
                    "ingestion_run_id": ingestion_run_id,
                }
            )

    return rows


def fetch_lineups_for_date(
    target_date: date,
    *,
    as_of: datetime | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Fetch and archive lineup plus starter observations for one MLB date.

    Workflow:
    1. Fetch and archive the day schedule with probable-pitcher hydration.
    2. Normalize one probable-starter row per team into ``starter_snapshot``.
    3. Fetch and archive every game's boxscore separately.
    4. Normalize official batting-order rows into ``lineup_snapshot``.

    An unavailable lineup produces no player rows. Consumers should determine
    availability at team/game level from the latest snapshot or from the
    companion return attributes described by their feature pipeline.
    """
    observed_at = as_of or datetime.now(UTC)
    _require_aware(observed_at)
    requested = target_date.isoformat()
    run_id = run_id or stable_id("ingestion", "mlb_lineups", observed_at.isoformat())

    schedule_params = {
        "sportId": 1,
        "date": requested,
        "hydrate": "probablePitcher",
    }
    schedule_payload = statsapi.get("schedule", schedule_params)
    games = _flatten_schedule_games(schedule_payload)
    schedule_hash = _persist_raw_payload(
        source=MLB_STARTER_SOURCE,
        body_object=schedule_payload,
        observed_at=observed_at,
        request_params=schedule_params,
        run_id=run_id,
    )

    starter_rows = [
        _starter_row(
            game=game,
            side=side,
            observed_at=observed_at,
            raw_payload_hash=schedule_hash,
            ingestion_run_id=run_id,
        )
        for game in games
        for side in ("away", "home")
    ]
    starter_frame = pd.DataFrame(starter_rows)
    if not starter_frame.empty:
        starter_frame = starter_frame.sort_values(["game_id", "side"]).reset_index(drop=True)
    _archive_frame(starter_frame, source="starter_snapshot", observed_at=observed_at)

    lineup_rows: list[dict[str, object]] = []
    for game in games:
        game_pk = int(game["gamePk"])
        boxscore_params = {"gamePk": game_pk}
        try:
            boxscore_payload = statsapi.get("game_boxscore", boxscore_params)
        except Exception:
            # Preserve the schedule/starter observation even when a game
            # boxscore is temporarily unavailable before the lineup is posted.
            continue

        boxscore_hash = _persist_raw_payload(
            source=MLB_LINEUP_SOURCE,
            body_object=boxscore_payload,
            observed_at=observed_at,
            request_params=boxscore_params,
            run_id=run_id,
        )
        lineup_rows.extend(
            _lineup_rows_from_boxscore(
                game=game,
                boxscore=boxscore_payload,
                observed_at=observed_at,
                raw_payload_hash=boxscore_hash,
                ingestion_run_id=run_id,
            )
        )

    lineup_frame = pd.DataFrame(lineup_rows)
    if not lineup_frame.empty:
        lineup_frame = (
            lineup_frame.sort_values(["game_id", "side", "batting_order", "player_id"])
            .drop_duplicates(
                subset=["game_id", "side", "player_id", "observed_at"],
                keep="last",
            )
            .reset_index(drop=True)
        )
    _archive_frame(lineup_frame, source="lineup_snapshot", observed_at=observed_at)
    return lineup_frame
