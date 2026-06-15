from __future__ import annotations

from collections import defaultdict
from time import monotonic

from app.storage.models import Detection, Point, Zone, ZoneRule


class ZoneEngine:
    def __init__(self, zones: list[Zone] | None = None) -> None:
        self.zones = zones or []
        self._inside_tracks: dict[tuple[str, str, int], bool] = {}
        self._last_centroids: dict[tuple[str, str, int], Point] = {}
        self._counted_tracks: set[tuple[str, str, int, str]] = set()
        self._pending_batches: dict[tuple[str, str], tuple[float, set[int]]] = {}
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def set_zones(self, zones: list[Zone]) -> None:
        self.zones = zones

    def contains(self, zone: Zone, point: Point) -> bool:
        x, y = point
        inside = False
        polygon = zone.polygon
        j = len(polygon) - 1
        for i, point_i in enumerate(polygon):
            xi, yi = point_i
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def annotate_detections(self, detections: list[Detection]) -> list[Detection]:
        for detection in detections:
            centroid = detection.centroid or self._bbox_centroid(detection.bbox)
            detection.centroid = centroid
            detection.zones = []
            for zone in self.zones:
                if not zone.enabled or not self.contains(zone, centroid):
                    self._mark_outside(zone, detection)
                    continue
                detection.zones.append(zone.name)
                self._handle_counting(zone, detection)
        return detections

    def restricted_violations(self, detections: list[Detection]) -> list[tuple[Zone, Detection]]:
        violations: list[tuple[Zone, Detection]] = []
        for detection in detections:
            centroid = detection.centroid or self._bbox_centroid(detection.bbox)
            for zone in self.zones:
                if zone.zone_type not in {"restricted", "danger"}:
                    continue
                if not zone.enabled or not self.contains(zone, centroid):
                    continue
                if not zone.rules:
                    violations.append((zone, detection))
                    continue
                for rule in zone.rules:
                    class_match = not rule.classes or detection.class_name in rule.classes
                    if class_match and rule.alert_on_entry:
                        violations.append((zone, detection))
        return violations

    def zone_occupancy(self, detections: list[Detection]) -> dict[str, int]:
        occupancy = {zone.name: 0 for zone in self.zones}
        for detection in detections:
            for zone_name in detection.zones:
                occupancy[zone_name] = occupancy.get(zone_name, 0) + 1
        return occupancy

    def _handle_counting(self, zone: Zone, detection: Detection) -> None:
        if detection.track_id is None:
            return
        for rule in zone.rules:
            if not rule.count_on_entry:
                continue
            if rule.classes and detection.class_name not in rule.classes:
                continue
            if rule.counting_line:
                self._handle_line_counting(zone, rule, detection)
            else:
                self._handle_entry_counting(zone, rule, detection)

    def _handle_entry_counting(self, zone: Zone, rule: ZoneRule, detection: Detection) -> None:
        if detection.track_id is None:
            return
        key = (zone.name, detection.class_name, detection.track_id)
        if not self._inside_tracks.get(key):
            self._record_count(zone, rule, detection)
        self._inside_tracks[key] = True

    def _handle_line_counting(self, zone: Zone, rule: ZoneRule, detection: Detection) -> None:
        if detection.track_id is None or detection.centroid is None or not rule.counting_line:
            return
        key = (zone.name, detection.class_name, detection.track_id)
        previous = self._last_centroids.get(key)
        current = detection.centroid
        self._last_centroids[key] = current
        if previous is None:
            return
        if not self._crossed_line(previous, current, rule.counting_line):
            return
        if not self._direction_matches(previous, current, rule.direction):
            return
        counted_key = (zone.name, detection.class_name, detection.track_id, str(rule.counting_line))
        if counted_key in self._counted_tracks:
            return
        self._counted_tracks.add(counted_key)
        self._record_count(zone, rule, detection)

    def _record_count(self, zone: Zone, rule: ZoneRule, detection: Detection) -> None:
        if detection.track_id is None:
            return
        target_count = max(1, int(rule.target_count))
        if rule.count_mode == "individual" and target_count <= 1:
            self.counts[zone.name][detection.class_name] += 1
            return
        batch_key = (zone.name, detection.class_name)
        now = monotonic()
        started_at, track_ids = self._pending_batches.get(batch_key, (now, set()))
        if now - started_at > rule.batch_window_seconds:
            started_at, track_ids = now, set()
        track_ids.add(detection.track_id)
        min_required = max(rule.batch_min_objects, target_count)
        max_allowed = rule.batch_max_objects
        if len(track_ids) >= min_required and (max_allowed is None or len(track_ids) <= max_allowed):
            self.counts[zone.name][detection.class_name] += len(track_ids)
            self._pending_batches[batch_key] = (now, set())
        else:
            self._pending_batches[batch_key] = (started_at, track_ids)

    def _mark_outside(self, zone: Zone, detection: Detection) -> None:
        if detection.track_id is None:
            return
        key = (zone.name, detection.class_name, detection.track_id)
        self._inside_tracks[key] = False

    def _bbox_centroid(self, bbox: list[float]) -> Point:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _direction_matches(self, previous: Point, current: Point, direction: str) -> bool:
        if direction == "any":
            return True
        if direction == "left_to_right":
            return current[0] > previous[0]
        if direction == "right_to_left":
            return current[0] < previous[0]
        return True

    def _crossed_line(self, previous: Point, current: Point, line: list[Point]) -> bool:
        if len(line) < 2:
            return False
        a, b = line[0], line[1]
        previous_side = self._side_of_line(previous, a, b)
        current_side = self._side_of_line(current, a, b)
        if previous_side == 0 or current_side == 0:
            return previous_side != current_side
        return (previous_side < 0 < current_side) or (current_side < 0 < previous_side)

    def _side_of_line(self, point: Point, line_start: Point, line_end: Point) -> float:
        return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (
            line_end[1] - line_start[1]
        ) * (point[0] - line_start[0])
