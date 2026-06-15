from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


Downloader = Callable[[str, str, str], str]


class HuggingFaceModelRegistry:
    def __init__(
        self,
        models_dir: Path,
        repo_id: str | None = None,
        downloader: Downloader | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.repo_id = repo_id
        self.downloader = downloader or self._default_download

    def resolve(self, model_name: str) -> Path:
        requested = Path(model_name)
        if requested.is_absolute() and requested.exists():
            return requested.resolve()

        local_path = (self.models_dir / requested.name).resolve()
        if local_path.exists():
            return local_path

        if not self.repo_id:
            return local_path

        self.models_dir.mkdir(parents=True, exist_ok=True)
        downloaded = self.downloader(self.repo_id, requested.name, str(self.models_dir))
        return Path(downloaded).resolve()

    def _default_download(self, repo_id: str, filename: str, local_dir: str) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("huggingface_hub is required to download model weights from Hugging Face") from exc
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
