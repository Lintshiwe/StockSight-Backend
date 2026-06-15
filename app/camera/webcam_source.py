from __future__ import annotations

import cv2


class WebcamSource:
    def __init__(self, index: int) -> None:
        self.index = index
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> cv2.VideoCapture:
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam index {self.index}")
        return self.capture

