from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from time import perf_counter

from app.storage.models import Detection


@dataclass
class AnalyticsSnapshot:
    fps: float = 0.0
    total_detected_packages: int = 0
    total_detected_pallets: int = 0
    active_forklifts: int = 0
    active_people: int = 0
    objects_per_class: dict[str, int] = field(default_factory=dict)
    zone_occupancy: dict[str, int] = field(default_factory=dict)
    detection_confidence_averages: dict[str, float] = field(default_factory=dict)


class AnalyticsEngine:
    def __init__(self, fps_window: int = 30) -> None:
        self.frame_times: deque[float] = deque(maxlen=fps_window)
        self.snapshot = AnalyticsSnapshot()

    def tick(self) -> float:
        now = perf_counter()
        self.frame_times.append(now)
        if len(self.frame_times) < 2:
            self.snapshot.fps = 0.0
            return 0.0
        duration = self.frame_times[-1] - self.frame_times[0]
        self.snapshot.fps = (len(self.frame_times) - 1) / duration if duration > 0 else 0.0
        return self.snapshot.fps

    def update(self, detections: list[Detection], zone_occupancy: dict[str, int] | None = None) -> AnalyticsSnapshot:
        counts = Counter(detection.class_name for detection in detections)
        confidences: dict[str, list[float]] = {}
        for detection in detections:
            confidences.setdefault(detection.class_name, []).append(detection.confidence)

        self.snapshot.objects_per_class = dict(counts)
        self.snapshot.total_detected_packages += counts.get("package", 0) + counts.get("box", 0)
        self.snapshot.total_detected_pallets += counts.get("pallet", 0)
        self.snapshot.active_forklifts = counts.get("forklift", 0)
        self.snapshot.active_people = counts.get("person", 0)
        self.snapshot.zone_occupancy = zone_occupancy or {}
        self.snapshot.detection_confidence_averages = {
            class_name: sum(values) / len(values) for class_name, values in confidences.items()
        }
        return self.snapshot

    def summary(self) -> dict[str, object]:
        return self.snapshot.__dict__

