from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buffer.tobytes()


def save_jpeg(frame: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"Failed to save image to {path}")
    return path

