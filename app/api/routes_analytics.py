from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request


router = APIRouter(prefix="/api/analytics")


def runtime(request: Request) -> Any:
    return request.app.state.runtime


@router.get("/summary")
def summary(request: Request) -> dict[str, Any]:
    return runtime(request).analytics.summary()


@router.get("/timeline")
def timeline(request: Request) -> list[dict[str, Any]]:
    events = request.app.state.repository.list_events(limit=200)
    return [{"timestamp": event.timestamp, "event_type": event.event_type, "severity": event.severity} for event in events]


@router.get("/classes")
def classes(request: Request) -> dict[str, Any]:
    return {"objects_per_class": runtime(request).analytics.snapshot.objects_per_class}


@router.get("/zones")
def zones(request: Request) -> dict[str, Any]:
    return {"zone_occupancy": runtime(request).analytics.snapshot.zone_occupancy}

