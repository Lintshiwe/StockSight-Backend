from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


Point = tuple[float, float]
ZoneType = Literal["counting", "restricted", "loading", "shelf", "danger", "aisle"]
CountMode = Literal["individual", "batch"]
CountDirection = Literal["left_to_right", "right_to_left", "any"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ZoneRule:
    classes: list[str] = field(default_factory=list)
    alert_on_entry: bool = False
    count_on_entry: bool = False
    dwell_seconds: float | None = None
    count_mode: CountMode = "individual"
    direction: CountDirection = "any"
    target_count: int = 1
    batch_window_seconds: float = 1.0
    batch_min_objects: int = 1
    batch_max_objects: int | None = None
    counting_line: list[Point] | None = None


@dataclass
class Zone:
    name: str
    zone_type: ZoneType
    polygon: list[Point]
    id: int | None = None
    rules: list[ZoneRule] = field(default_factory=list)
    enabled: bool = True
    created_at: str = field(default_factory=utc_now)


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list[float]
    mask: list[Point] | None = None
    area: float = 0.0
    track_id: int | None = None
    centroid: Point | None = None
    zones: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now)


@dataclass
class EventCreate:
    event_type: str
    severity: str
    message: str
    class_name: str | None = None
    zone_name: str | None = None
    track_id: int | None = None
    confidence: float | None = None
    screenshot_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event(EventCreate):
    id: int | None = None
    timestamp: str = field(default_factory=utc_now)
