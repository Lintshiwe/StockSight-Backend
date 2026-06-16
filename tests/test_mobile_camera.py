import asyncio
from types import SimpleNamespace

import cv2
import numpy as np

from app.api import routes_camera
from app.camera.camera_manager import CameraManager


def test_mobile_camera_ingests_jpeg_frame() -> None:
    manager = CameraManager("mobile")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)

    assert ok
    status = manager.ingest_jpeg(encoded.tobytes())
    latest = manager.read_latest()

    assert status["running"] is True
    assert status["source"] == "mobile"
    assert status["source_type"] == "mobile"
    assert status["width"] == 64
    assert status["height"] == 48
    assert latest is not None
    assert latest.shape[:2] == (48, 64)


def test_mobile_camera_frame_endpoint_updates_annotated_stream() -> None:
    from app.storage.models import Detection

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok

    class FakeCamera:
        def __init__(self) -> None:
            self.status_payload = {
                "running": True,
                "source": "mobile",
                "source_type": "mobile",
                "width": 64,
                "height": 48,
                "last_error": None,
            }

        def ingest_jpeg(self, data: bytes) -> dict[str, object]:
            return self.status_payload

        def status(self) -> dict[str, object]:
            return self.status_payload

    class FakeRuntime:
        def __init__(self) -> None:
            self.camera = FakeCamera()
            self.latest_frame_jpeg: bytes | None = None

        def process_mobile_frame(self, decoded_frame, confidence=None, iou=None):
            self.latest_frame_jpeg = encoded.tobytes()
            return [Detection(class_name="box", confidence=0.9, bbox=[1, 2, 20, 30], area=500)]

    fake_runtime = FakeRuntime()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(runtime=fake_runtime)),
        body=lambda: asyncio.sleep(0, result=encoded.tobytes()),
    )

    response = asyncio.run(routes_camera.mobile_camera_frame(request))

    assert response["processed"] is True
    assert response["detections"] == 1
    assert fake_runtime.latest_frame_jpeg == encoded.tobytes()
