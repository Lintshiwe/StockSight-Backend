from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


router = APIRouter(prefix="/api/model")


class LoadModelRequest(BaseModel):
    model_path: str | None = None


class ModelSettingsRequest(BaseModel):
    confidence_threshold: float | None = None
    iou_threshold: float | None = None
    image_size: int | None = None
    frame_skip: int | None = None
    prefer_gpu: bool | None = None
    enable_tracking: bool | None = None
    enable_masks: bool | None = None
    enable_zone_alerts: bool | None = None
    detection_mode: str | None = None
    allowed_classes: list[str] | None = None
    blocked_classes: list[str] | None = None


def runtime(request: Request) -> Any:
    return request.app.state.runtime


@router.get("/status")
def model_status(request: Request) -> dict[str, Any]:
    return runtime(request).model_status()


@router.post("/load")
def load_model(payload: LoadModelRequest, request: Request) -> dict[str, Any]:
    try:
        return runtime(request).load_model(payload.model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/settings")
def model_settings(payload: ModelSettingsRequest, request: Request) -> dict[str, Any]:
    updates = {key: value for key, value in payload.dict().items() if value is not None}
    return runtime(request).update_model_settings(updates)
