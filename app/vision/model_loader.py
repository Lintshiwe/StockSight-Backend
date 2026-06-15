from __future__ import annotations

from pathlib import Path
from typing import Any

from app.utils.logger import get_logger


logger = get_logger(__name__)


class ModelLoader:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_path: Path | None = None
        self.device = "cpu"
        self.cuda_available = False
        self.class_names: dict[int, str] = {}

    def load(self, model_path: Path, prefer_gpu: bool = True) -> dict[str, Any]:
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Ultralytics and PyTorch are required for live inference") from exc

        self.cuda_available = bool(torch.cuda.is_available())
        self.device = "cuda:0" if prefer_gpu and self.cuda_available else "cpu"
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.model_path = model_path
        self.class_names = dict(getattr(self.model, "names", {}) or {})
        logger.info("Loaded YOLO model %s on %s", model_path, self.device)
        return self.status()

    def warmup(self, image_size: int = 640) -> None:
        if self.model is None:
            return
        import numpy as np

        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, imgsz=image_size, device=self.device, verbose=False)
        except AttributeError as exc:
            logger.warning("Model warmup skipped after non-fatal Ultralytics callback error: %s", exc)

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "model_path": str(self.model_path) if self.model_path else None,
            "device": self.device,
            "cuda_available": self.cuda_available,
            "class_names": self.class_names,
        }
