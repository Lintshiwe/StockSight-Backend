from app.storage.models import Detection, Zone, ZoneRule
from app.vision.zone_engine import ZoneEngine


def make_counter(direction: str, target_count: int = 1) -> ZoneEngine:
    zone = Zone(
        id=1,
        name="Retail Counter",
        zone_type="counting",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        rules=[
            ZoneRule(
                classes=["package", "couch"],
                count_on_entry=True,
                count_mode="individual",
                direction=direction,
                target_count=target_count,
                counting_line=[(50, 0), (50, 100)],
            )
        ],
    )
    return ZoneEngine([zone])


def detection(track_id: int, x: float, class_name: str = "package") -> Detection:
    return Detection(class_name, 0.9, [x - 5, 20, x + 5, 40], centroid=(x, 30), track_id=track_id)


def test_left_to_right_counter_ignores_right_to_left_motion() -> None:
    engine = make_counter("left_to_right")

    engine.annotate_detections([detection(1, 60)])
    engine.annotate_detections([detection(1, 40)])

    assert engine.counts["Retail Counter"]["package"] == 0


def test_left_to_right_counter_counts_crossing_once() -> None:
    engine = make_counter("left_to_right")

    engine.annotate_detections([detection(1, 40)])
    engine.annotate_detections([detection(1, 60)])
    engine.annotate_detections([detection(1, 80)])

    assert engine.counts["Retail Counter"]["package"] == 1


def test_target_count_two_requires_two_distinct_tracks() -> None:
    engine = make_counter("left_to_right", target_count=2)

    engine.annotate_detections([detection(1, 40), detection(2, 42)])
    engine.annotate_detections([detection(1, 60)])
    assert engine.counts["Retail Counter"]["package"] == 0

    engine.annotate_detections([detection(2, 62)])
    assert engine.counts["Retail Counter"]["package"] == 2

