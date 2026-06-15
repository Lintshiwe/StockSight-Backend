from __future__ import annotations

import json
from typing import Any

from .database import Database
from .models import Event, EventCreate, Zone, ZoneRule, utc_now


class Repository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create_event(self, event: EventCreate) -> Event:
        timestamp = utc_now()
        with self.database.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    timestamp, event_type, severity, message, class_name, zone_name,
                    track_id, confidence, screenshot_path, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    event.event_type,
                    event.severity,
                    event.message,
                    event.class_name,
                    event.zone_name,
                    event.track_id,
                    event.confidence,
                    event.screenshot_path,
                    json.dumps(event.metadata),
                ),
            )
            event_id = int(cursor.lastrowid)
        return Event(**event.__dict__, id=event_id, timestamp=timestamp)

    def get_event(self, event_id: int) -> Event | None:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._event_from_row(row) if row else None

    def list_events(self, limit: int = 100, event_type: str | None = None) -> list[Event]:
        query = "SELECT * FROM events"
        params: list[Any] = []
        if event_type:
            query += " WHERE event_type = ?"
            params.append(event_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self.database.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._event_from_row(row) for row in rows]

    def delete_event(self, event_id: int) -> bool:
        with self.database.connect() as conn:
            cursor = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            return cursor.rowcount > 0

    def create_zone(self, zone: Zone) -> Zone:
        with self.database.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO zones (name, zone_type, polygon, rules, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    zone.name,
                    zone.zone_type,
                    json.dumps(zone.polygon),
                    json.dumps([rule.__dict__ for rule in zone.rules]),
                    1 if zone.enabled else 0,
                    zone.created_at,
                ),
            )
            zone.id = int(cursor.lastrowid)
        return zone

    def update_zone(self, zone_id: int, zone: Zone) -> Zone | None:
        with self.database.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE zones
                SET name = ?, zone_type = ?, polygon = ?, rules = ?, enabled = ?
                WHERE id = ?
                """,
                (
                    zone.name,
                    zone.zone_type,
                    json.dumps(zone.polygon),
                    json.dumps([rule.__dict__ for rule in zone.rules]),
                    1 if zone.enabled else 0,
                    zone_id,
                ),
            )
        if cursor.rowcount == 0:
            return None
        zone.id = zone_id
        return zone

    def list_zones(self) -> list[Zone]:
        with self.database.connect() as conn:
            rows = conn.execute("SELECT * FROM zones ORDER BY id").fetchall()
        return [self._zone_from_row(row) for row in rows]

    def delete_zone(self, zone_id: int) -> bool:
        with self.database.connect() as conn:
            cursor = conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
            return cursor.rowcount > 0

    def _event_from_row(self, row: Any) -> Event:
        return Event(
            id=row["id"],
            timestamp=row["timestamp"],
            event_type=row["event_type"],
            severity=row["severity"],
            message=row["message"],
            class_name=row["class_name"],
            zone_name=row["zone_name"],
            track_id=row["track_id"],
            confidence=row["confidence"],
            screenshot_path=row["screenshot_path"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def _zone_from_row(self, row: Any) -> Zone:
        rules = [ZoneRule(**item) for item in json.loads(row["rules"] or "[]")]
        polygon = [tuple(point) for point in json.loads(row["polygon"])]
        return Zone(
            id=row["id"],
            name=row["name"],
            zone_type=row["zone_type"],
            polygon=polygon,
            rules=rules,
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
        )

