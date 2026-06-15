from __future__ import annotations

from app.storage.models import Detection, EventCreate, Zone


class AlertEngine:
    def restricted_zone_events(self, violations: list[tuple[Zone, Detection]]) -> list[EventCreate]:
        events: list[EventCreate] = []
        seen: set[tuple[str, int | None, str]] = set()
        for zone, detection in violations:
            key = (zone.name, detection.track_id, detection.class_name)
            if key in seen:
                continue
            seen.add(key)
            events.append(
                EventCreate(
                    event_type="restricted_zone_entry",
                    severity="high" if zone.zone_type == "restricted" else "critical",
                    message=f"{detection.class_name} entered {zone.name}",
                    class_name=detection.class_name,
                    zone_name=zone.name,
                    track_id=detection.track_id,
                    confidence=detection.confidence,
                    metadata={"zones": detection.zones, "bbox": detection.bbox},
                )
            )
        return events

    def safety_violation_events(self, detections: list[Detection]) -> list[EventCreate]:
        classes = {d.class_name for d in detections}
        if "person" not in classes:
            return []
        events: list[EventCreate] = []
        if "helmet" not in classes:
            events.append(EventCreate("ppe_violation", "medium", "Person detected without visible helmet"))
        if "safety_vest" not in classes:
            events.append(EventCreate("ppe_violation", "medium", "Person detected without visible safety vest"))
        return events

