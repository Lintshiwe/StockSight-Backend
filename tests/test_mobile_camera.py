import cv2
import numpy as np

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
