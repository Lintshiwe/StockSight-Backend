from __future__ import annotations

from urllib.parse import urlparse

import cv2


ALLOWED_SCHEMES = {"rtsp", "http", "https"}


class RTSPSource:
    def __init__(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ALLOWED_SCHEMES or not parsed.netloc:
            raise ValueError("Camera URL must be an RTSP, HTTP, or HTTPS URL")
        self.url = url
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> cv2.VideoCapture:
        self.capture = cv2.VideoCapture(self.url)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera stream {self.url}")
        return self.capture

