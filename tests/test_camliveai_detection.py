from app.main import app


def test_camliveai_detection_route_is_registered() -> None:
    assert any(route.path == "/detection/single" for route in app.routes)
