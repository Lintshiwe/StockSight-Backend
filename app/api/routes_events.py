from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(prefix="/api/events")


def repo(request: Request) -> Any:
    return request.app.state.repository


@router.get("")
def list_events(request: Request, limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
    return [asdict(event) for event in repo(request).list_events(limit=limit, event_type=event_type)]


@router.get("/{event_id}")
def get_event(event_id: int, request: Request) -> dict[str, Any]:
    event = repo(request).get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return asdict(event)


@router.delete("/{event_id}")
def delete_event(event_id: int, request: Request) -> dict[str, bool]:
    deleted = repo(request).delete_event(event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"deleted": True}

