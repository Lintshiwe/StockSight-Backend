from app.main import app, health


def test_health_endpoint() -> None:
    assert any(route.path == "/health" for route in app.routes)
    assert health()["status"] == "ok"
