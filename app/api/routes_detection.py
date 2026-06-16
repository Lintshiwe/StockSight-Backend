from __future__ import annotations

import base64
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from app.config import settings


router = APIRouter()


class SingleDetectionRequest(BaseModel):
    image: str
    confidence: float | None = None
    iou: float | None = None


def runtime(request: Request) -> Any:
    return request.app.state.runtime


@router.post("/detection/single")
def detection_single(
    payload: SingleDetectionRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    expected = f"Bearer {settings.camliveai_token}"
    if settings.camliveai_token and authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid CamLiveAI token")

    started = time.perf_counter()
    image_data = payload.image.split(",", 1)[1] if payload.image.startswith("data:") else payload.image
    try:
        raw = base64.b64decode(image_data, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode JPEG image")

    service = runtime(request)
    detections = service.detect_frame(frame, payload.confidence, payload.iou)
    response_detections = []
    class_lookup = {name: class_id for class_id, name in service.model_loader.class_names.items()}
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        response_detections.append(
            {
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "class_id": class_lookup.get(detection.class_name),
            }
        )

    return {
        "detections": response_detections,
        "total_count": len(response_detections),
        "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        "model_used": service.model_loader.model_path.name if service.model_loader.model_path else "loading",
        "tenant_type": "warehouse",
    }
