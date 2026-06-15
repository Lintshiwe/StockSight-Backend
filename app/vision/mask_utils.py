from __future__ import annotations

import cv2
import numpy as np

from app.storage.models import Detection, Point, Zone


def bbox_centroid(bbox: list[float]) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def polygon_area(points: list[Point] | None) -> float:
    if not points or len(points) < 3:
        return 0.0
    contour = np.array(points, dtype=np.float32)
    return float(abs(cv2.contourArea(contour)))


def draw_annotations(frame: np.ndarray, detections: list[Detection], zones: list[Zone]) -> np.ndarray:
    annotated = frame.copy()
    for zone in zones:
        pts = np.array(zone.polygon, dtype=np.int32)
        if len(pts) >= 3:
            color = (38, 120, 204) if zone.zone_type == "restricted" else (93, 184, 114)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(annotated, zone.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    overlay = annotated.copy()
    for detection in detections:
        color = class_color(detection.class_name)
        if detection.mask and len(detection.mask) >= 3:
            pts = np.array(detection.mask, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        track = f" #{detection.track_id}" if detection.track_id is not None else ""
        label = f"{detection.class_name}{track} {detection.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    annotated = cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0)
    return annotated


def class_color(class_name: str) -> tuple[int, int, int]:
    palette = {
        "person": (198, 69, 69),
        "forklift": (232, 165, 90),
        "package": (93, 184, 166),
        "box": (93, 184, 166),
        "pallet": (212, 160, 23),
        "helmet": (93, 184, 114),
        "safety_vest": (93, 184, 114),
    }
    return palette.get(class_name, (204, 120, 92))

