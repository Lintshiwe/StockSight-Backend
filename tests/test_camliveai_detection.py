import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api import routes_detection
from app.storage.models import Detection


def encoded_test_image() -> str:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def request_for(runtime: object | None = None) -> object:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=runtime or SimpleNamespace())))


def test_camliveai_detection_route_is_registered() -> None:
    assert any(route.path == "/detection/single" for route in routes_detection.router.routes)


def test_camliveai_detection_route_is_registered_on_app() -> None:
    from app.main import app

    assert any(route.path == "/detection/single" for route in app.routes)


def test_camliveai_detection_rejects_missing_image() -> None:
    with pytest.raises(ValidationError):
        routes_detection.SingleDetectionRequest.model_validate({})


def test_camliveai_detection_rejects_bad_base64() -> None:
    payload = routes_detection.SingleDetectionRequest(image="not-valid-base64")

    with pytest.raises(HTTPException) as exc:
        routes_detection.detection_single(
            payload,
            request_for(),
            authorization="Bearer stocksight-camliveai-token",
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid base64 image"


def test_camliveai_detection_rejects_invalid_token() -> None:
    payload = routes_detection.SingleDetectionRequest(image=encoded_test_image())

    with pytest.raises(HTTPException) as exc:
        routes_detection.detection_single(payload, request_for(), authorization="Bearer wrong")

    assert exc.value.status_code == 401


def test_camliveai_detection_returns_503_when_model_unavailable() -> None:
    camera = SimpleNamespace(ingest_jpeg=lambda data: None)
    runtime = SimpleNamespace(
        camera=camera,
        process_mobile_frame=lambda frame, confidence=None, iou=None: (_ for _ in ()).throw(
            RuntimeError("Detection model is not available")
        ),
    )
    payload = routes_detection.SingleDetectionRequest(image=encoded_test_image())

    with pytest.raises(HTTPException) as exc:
        routes_detection.detection_single(
            payload,
            request_for(runtime),
            authorization="Bearer stocksight-camliveai-token",
        )

    assert exc.value.status_code == 503
    assert exc.value.detail == "Detection model is not available"


def test_camliveai_detection_maps_stock_sight_detections() -> None:
    camera = SimpleNamespace(ingest_jpeg=lambda data: None)
    model_loader = SimpleNamespace(class_names={0: "box"}, model_path=SimpleNamespace(name="warehouse.pt"))
    runtime = SimpleNamespace(
        camera=camera,
        model_loader=model_loader,
        settings=SimpleNamespace(model_path="warehouse.pt"),
        process_mobile_frame=lambda frame, confidence=None, iou=None: [
            Detection(
                class_name="box",
                confidence=0.91,
                bbox=[10.0, 20.0, 50.0, 70.0],
                area=2000.0,
                centroid=(30.0, 45.0),
            )
        ],
    )
    payload = routes_detection.SingleDetectionRequest(
        image=encoded_test_image(),
        confidence=0.4,
        iou=0.6,
    )

    body = routes_detection.detection_single(
        payload,
        request_for(runtime),
        authorization="Bearer stocksight-camliveai-token",
    )

    assert body["detections"] == [
        {
            "bbox": [10.0, 20.0, 40.0, 50.0],
            "class_name": "box",
            "confidence": 0.91,
            "class_id": 0,
        }
    ]
    assert body["total_count"] == 1
    assert body["model_used"] == "warehouse.pt"
    assert body["tenant_type"] == "retail"
