from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Any

import cv2
import numpy as np

from app.camera.rtsp_source import RTSPSource
from app.camera.webcam_source import WebcamSource
from app.utils.logger import get_logger


logger = get_logger(__name__)


class CameraManager:
    def __init__(self, source: str = "0", queue_size: int = 2, reconnect_seconds: float = 2.0) -> None:
        self.source = source
        self.reconnect_seconds = reconnect_seconds
        self.frames: Queue[np.ndarray] = Queue(maxsize=queue_size)
        self.capture: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.last_error: str | None = None
        self.frame_width = 0
        self.frame_height = 0
        self.source_type = self._source_type(source)

    def set_source(self, source: str) -> None:
        was_running = self.running
        if was_running:
            self.stop()
        self.source = source
        self.source_type = self._source_type(source)
        if was_running:
            self.start()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        if self.source_type == "mobile":
            self.last_error = None
            return
        self.thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        self.thread = None
        if self.capture:
            self.capture.release()
        self.capture = None

    def ingest_jpeg(self, data: bytes) -> dict[str, Any]:
        if self.source_type != "mobile":
            self.set_source("mobile")
        if not self.running:
            self.start()
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            self.last_error = "Unable to decode mobile camera frame"
            raise ValueError(self.last_error)
        self.frame_height, self.frame_width = frame.shape[:2]
        self.last_error = None
        if self.frames.full():
            self.frames.get_nowait()
        self.frames.put_nowait(frame)
        return self.status()

    def read_latest(self) -> np.ndarray | None:
        latest: np.ndarray | None = None
        while not self.frames.empty():
            latest = self.frames.get_nowait()
        return latest

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "source": self.source,
            "source_type": self.source_type,
            "width": self.frame_width,
            "height": self.frame_height,
            "last_error": self.last_error,
        }

    def _open_capture(self) -> cv2.VideoCapture:
        source = WebcamSource(int(self.source)) if self.source.isdigit() else RTSPSource(self.source)
        capture = source.open()
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return capture

    def _source_type(self, source: str) -> str:
        if source == "mobile":
            return "mobile"
        return "webcam" if source.isdigit() else "stream"

    def _capture_loop(self) -> None:
        while self.running:
            try:
                if self.capture is None or not self.capture.isOpened():
                    self.capture = self._open_capture()
                    self.last_error = None
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera read failed")
                if self.frames.full():
                    self.frames.get_nowait()
                self.frames.put_nowait(frame)
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.warning("Camera error: %s", exc)
                if self.capture:
                    self.capture.release()
                    self.capture = None
                time.sleep(self.reconnect_seconds)
