from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.storage.models import Zone, ZoneRule


router = APIRouter(prefix="/api/zones")


class ZoneRuleRequest(BaseModel):
    classes: list[str] = Field(default_factory=list)
    alert_on_entry: bool = False
    count_on_entry: bool = False
    dwell_seconds: float | None = None
    count_mode: Literal["individual", "batch"] = "individual"
    direction: Literal["left_to_right", "right_to_left", "any"] = "any"
    target_count: int = 1
    batch_window_seconds: float = 1.0
    batch_min_objects: int = 1
    batch_max_objects: int | None = None
    counting_line: list[tuple[float, float]] | None = None


class ZoneRequest(BaseModel):
    name: str
    zone_type: Literal["counting", "restricted", "loading", "shelf", "danger", "aisle"]
    polygon: list[tuple[float, float]]
    rules: list[ZoneRuleRequest] = Field(default_factory=list)
    enabled: bool = True


def repo(request: Request) -> Any:
    return request.app.state.repository


def runtime(request: Request) -> Any:
    return request.app.state.runtime


def to_zone(payload: ZoneRequest, zone_id: int | None = None) -> Zone:
    return Zone(
        id=zone_id,
        name=payload.name,
        zone_type=payload.zone_type,
        polygon=payload.polygon,
        rules=[ZoneRule(**rule.dict()) for rule in payload.rules],
        enabled=payload.enabled,
    )


@router.get("")
def list_zones(request: Request) -> list[dict[str, Any]]:
    return [asdict(zone) for zone in repo(request).list_zones()]


@router.post("")
def create_zone(payload: ZoneRequest, request: Request) -> dict[str, Any]:
    zone = repo(request).create_zone(to_zone(payload))
    runtime(request).refresh_zones()
    return asdict(zone)


@router.put("/{zone_id}")
def update_zone(zone_id: int, payload: ZoneRequest, request: Request) -> dict[str, Any]:
    zone = repo(request).update_zone(zone_id, to_zone(payload, zone_id))
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found")
    runtime(request).refresh_zones()
    return asdict(zone)


@router.delete("/{zone_id}")
def delete_zone(zone_id: int, request: Request) -> dict[str, bool]:
    deleted = repo(request).delete_zone(zone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    runtime(request).refresh_zones()
    return {"deleted": True}
