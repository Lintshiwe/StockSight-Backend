from pathlib import Path

from app.config import Settings


def test_settings_defaults_are_project_local(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)

    assert settings.database_path == tmp_path / "backend" / "data" / "warehousesight.db"
    assert settings.models_dir == tmp_path / "models"
    assert settings.camera_source == "0"
    assert settings.confidence_threshold > 0
    assert settings.iou_threshold > 0


def test_model_path_validation_stays_inside_models_dir(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    model = settings.safe_model_path("custom.pt")

    assert model == tmp_path / "models" / "custom.pt"

