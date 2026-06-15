from app.storage.models import Detection, Zone, ZoneRule
from app.vision.zone_engine import ZoneEngine


def test_point_in_polygon_and_zone_membership() -> None:
    zone = Zone(
        id=1,
        name="Restricted Dock",
        zone_type="restricted",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        rules=[ZoneRule(classes=["person"], alert_on_entry=True)],
    )
    engine = ZoneEngine([zone])

    detection = Detection(
        class_name="person",
        confidence=0.91,
        bbox=[10, 10, 30, 40],
        centroid=(20, 25),
        area=600,
        track_id=42,
    )

    memberships = engine.annotate_detections([detection])

    assert memberships[0].zones == ["Restricted Dock"]
    assert engine.contains(zone, (50, 50)) is True
    assert engine.contains(zone, (150, 50)) is False


def test_counting_zone_counts_track_once_per_entry() -> None:
    zone = Zone(
        id=2,
        name="Outbound Count",
        zone_type="counting",
        polygon=[(0, 0), (50, 0), (50, 50), (0, 50)],
        rules=[ZoneRule(classes=["package"], count_on_entry=True)],
    )
    engine = ZoneEngine([zone])

    outside = Detection("package", 0.9, [60, 10, 80, 30], None, 400, 7)
    inside = Detection("package", 0.9, [10, 10, 30, 30], None, 400, 7)

    engine.annotate_detections([outside])
    engine.annotate_detections([inside])
    engine.annotate_detections([inside])

    assert engine.counts["Outbound Count"]["package"] == 1

