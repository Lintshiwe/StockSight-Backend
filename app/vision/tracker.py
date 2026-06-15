from __future__ import annotations

from math import dist

from app.storage.models import Detection


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_missing: int = 20) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: dict[int, tuple[float, float]] = {}
        self.missing: dict[int, int] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_tracks = set(self.tracks)
        for detection in detections:
            if detection.track_id is not None:
                continue
            centroid = detection.centroid
            if centroid is None:
                continue
            best_id: int | None = None
            best_distance = self.max_distance
            for track_id in list(unmatched_tracks):
                distance = dist(centroid, self.tracks[track_id])
                if distance < best_distance:
                    best_id = track_id
                    best_distance = distance
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            detection.track_id = best_id
            self.tracks[best_id] = centroid
            self.missing[best_id] = 0
            unmatched_tracks.discard(best_id)

        for track_id in unmatched_tracks:
            self.missing[track_id] = self.missing.get(track_id, 0) + 1
            if self.missing[track_id] > self.max_missing:
                self.tracks.pop(track_id, None)
                self.missing.pop(track_id, None)
        return detections

