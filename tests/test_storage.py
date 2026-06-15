from pathlib import Path

from app.storage.database import Database
from app.storage.models import EventCreate, Zone
from app.storage.repository import Repository


def test_repository_event_and_zone_crud(tmp_path: Path) -> None:
    db = Database(tmp_path / "events.db")
    db.initialize()
    repo = Repository(db)

    event = repo.create_event(
        EventCreate(
            event_type="restricted_zone_entry",
            severity="high",
            message="Person entered restricted dock",
            class_name="person",
            zone_name="Restricted Dock",
            track_id=11,
            confidence=0.88,
            screenshot_path=None,
            metadata={"fps": 24.1},
        )
    )

    assert event.id is not None
    assert repo.get_event(event.id).message == "Person entered restricted dock"
    assert len(repo.list_events(limit=10)) == 1

    zone = repo.create_zone(
        Zone(
            name="Dock",
            zone_type="restricted",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        )
    )
    assert zone.id is not None
    assert repo.list_zones()[0].name == "Dock"

    assert repo.delete_event(event.id) is True
    assert repo.get_event(event.id) is None

