from pathlib import Path

from app.utils.hf_models import HuggingFaceModelRegistry


def test_resolve_existing_local_model_without_download(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "retail.pt"
    model_path.write_bytes(b"weights")

    registry = HuggingFaceModelRegistry(models_dir=models_dir)

    assert registry.resolve("retail.pt") == model_path.resolve()


def test_download_is_used_when_repo_configured_and_file_missing(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_download(repo_id: str, filename: str, local_dir: str) -> str:
        calls.append((repo_id, filename, local_dir))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")
        return str(path)

    registry = HuggingFaceModelRegistry(
        models_dir=tmp_path / "models",
        repo_id="user/Modle_V2.0",
        downloader=fake_download,
    )

    resolved = registry.resolve("retail.pt")

    assert resolved == (tmp_path / "models" / "retail.pt").resolve()
    assert calls == [("user/Modle_V2.0", "retail.pt", str(tmp_path / "models"))]

