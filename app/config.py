from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    project_root: Path = field(default_factory=_project_root)
    app_name: str = "WarehouseSight-AI"
    environment: str = "development"
    camera_source: str = "0"
    camera_reconnect_seconds: float = 2.0
    model_path: str = "yolo11n-seg.pt"
    hf_model_repo_id: str = ""
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    image_size: int = 640
    frame_skip: int = 1
    prefer_gpu: bool = True
    enable_tracking: bool = True
    enable_masks: bool = True
    enable_zone_alerts: bool = True
    detection_mode: str = "all_objects"
    allowed_classes: str = ""
    blocked_classes: str = ""
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    camliveai_token: str = "stocksight-camliveai-token"

    def __post_init__(self) -> None:
        self.project_root = Path(os.getenv("PROJECT_ROOT", str(self.project_root))).resolve()
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        self.camera_source = os.getenv("CAMERA_SOURCE", self.camera_source)
        self.model_path = os.getenv("MODEL_PATH", self.model_path)
        self.hf_model_repo_id = os.getenv("HF_MODEL_REPO_ID", self.hf_model_repo_id)
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", self.confidence_threshold))
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", self.iou_threshold))
        self.image_size = int(os.getenv("IMAGE_SIZE", self.image_size))
        self.frame_skip = max(1, int(os.getenv("FRAME_SKIP", self.frame_skip)))
        self.prefer_gpu = os.getenv("PREFER_GPU", str(self.prefer_gpu)).lower() in {"1", "true", "yes"}
        self.enable_tracking = os.getenv("ENABLE_TRACKING", str(self.enable_tracking)).lower() in {"1", "true", "yes"}
        self.enable_masks = os.getenv("ENABLE_MASKS", str(self.enable_masks)).lower() in {"1", "true", "yes"}
        self.enable_zone_alerts = os.getenv("ENABLE_ZONE_ALERTS", str(self.enable_zone_alerts)).lower() in {"1", "true", "yes"}
        self.detection_mode = os.getenv("DETECTION_MODE", self.detection_mode)
        self.allowed_classes = os.getenv("ALLOWED_CLASSES", self.allowed_classes)
        self.blocked_classes = os.getenv("BLOCKED_CLASSES", self.blocked_classes)
        self.cors_origins = os.getenv("CORS_ORIGINS", self.cors_origins)
        self.camliveai_token = os.getenv("MOBILE_API_TOKEN", os.getenv("CAMLIVEAI_TOKEN", self.camliveai_token))

    @property
    def backend_dir(self) -> Path:
        return self.project_root / "backend"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def datasets_dir(self) -> Path:
        return self.project_root / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs"

    @property
    def data_dir(self) -> Path:
        return self.backend_dir / "data"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "warehousesight.db"

    @property
    def screenshot_dir(self) -> Path:
        return self.data_dir / "screenshots"

    @property
    def cors_origin_list(self) -> list[str]:
        required = ["https://stocksight-frontend.netlify.app", "https://camliveai.netlify.app"]
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return list(dict.fromkeys([*origins, *required]))

    @property
    def allowed_class_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_classes.split(",") if item.strip()]

    @property
    def blocked_class_list(self) -> list[str]:
        return [item.strip() for item in self.blocked_classes.split(",") if item.strip()]

    def safe_model_path(self, requested_path: str | None = None) -> Path:
        raw = requested_path or self.model_path
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.models_dir / candidate
        candidate = candidate.resolve()
        models_root = self.models_dir.resolve()
        if models_root not in candidate.parents and candidate != models_root:
            raise ValueError("Model path must stay inside the project models directory")
        return candidate


settings = Settings()
