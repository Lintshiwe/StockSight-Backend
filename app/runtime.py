from __future__ import annotations

import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from app.camera.camera_manager import CameraManager
from app.config import Settings, settings
from app.storage.models import Detection
from app.storage.repository import Repository
from app.utils.image_io import encode_jpeg, save_jpeg
from app.utils.hf_models import HuggingFaceModelRegistry
from app.utils.logger import get_logger
from app.vision.alert_engine import AlertEngine
from app.vision.analytics_engine import AnalyticsEngine
from app.vision.detection_filter import DetectionFilter, DetectionFilterSettings
from app.vision.mask_utils import draw_annotations
from app.vision.model_loader import ModelLoader
from app.vision.segmenter import Segmenter
from app.vision.zone_engine import ZoneEngine


logger = get_logger(__name__)


class VisionRuntime:
    def __init__(self, repository: Repository, config: Settings = settings) -> None:
        self.settings = config
        self.repository = repository
        self.camera = CameraManager(config.camera_source, reconnect_seconds=config.camera_reconnect_seconds)
        self.model_loader = ModelLoader()
        self.model_registry = HuggingFaceModelRegistry(config.models_dir, config.hf_model_repo_id or None)
        self.segmenter = Segmenter(self.model_loader)
        self.detection_filter = DetectionFilter(self._filter_settings())
        self.zone_engine = ZoneEngine(repository.list_zones() if config.database_path.exists() else [])
        self.analytics = AnalyticsEngine()
        self.alerts = AlertEngine()
        self.processing_thread: threading.Thread | None = None
        self.processing = False
        self.latest_frame_jpeg: bytes | None = None
        self.latest_payload: dict[str, Any] = {"detections": [], "alerts": [], "fps": 0.0}
        self.last_error: str | None = None
        self.model_loading = False
        self._model_lock = threading.Lock()
        self._frame_index = 0

    def start_camera(self) -> dict[str, Any]:
        self.camera.start()
        self.processing = True
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_loop, name="vision-pipeline", daemon=True)
            self.processing_thread.start()
        return self.camera.status()

    def stop_camera(self) -> dict[str, Any]:
        self.processing = False
        self.camera.stop()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3)
        return self.camera.status()

    def set_camera_source(self, source: str) -> dict[str, Any]:
        self.camera.set_source(source)
        self.settings.camera_source = source
        return self.camera.status()

    def load_model(self, model_path: str | None = None) -> dict[str, Any]:
        if self.model_loading:
            return self.model_status()
        requested = model_path or self.settings.model_path
        if self.model_loader.model is not None and self.model_loader.model_path and self.model_loader.model_path.name == Path(requested).name:
            return self.model_status()
        if not self._model_lock.acquire(blocking=False):
            return self.model_status()
        self.model_loading = True
        try:
            path = self.model_registry.resolve(requested)
            self.settings.safe_model_path(path.name)
            status = self.model_loader.load(path, self.settings.prefer_gpu)
            self.model_loader.warmup(self.settings.image_size)
            return status
        finally:
            self.model_loading = False
            self._model_lock.release()

    def try_load_configured_model(self) -> None:
        try:
            self.load_model(self.settings.model_path)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Model auto-load failed: {exc}"
            logger.warning("Model auto-load failed: %s", exc)

    def update_model_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "confidence_threshold",
            "iou_threshold",
            "image_size",
            "frame_skip",
            "prefer_gpu",
            "enable_tracking",
            "enable_masks",
            "enable_zone_alerts",
            "detection_mode",
            "allowed_classes",
            "blocked_classes",
        ):
            if key in payload:
                value = payload[key]
                if key in {"allowed_classes", "blocked_classes"} and isinstance(value, list):
                    value = ",".join(value)
                setattr(self.settings, key, value)
        self.settings.frame_skip = max(1, int(self.settings.frame_skip))
        self.detection_filter = DetectionFilter(self._filter_settings())
        return self.model_status()

    def model_status(self) -> dict[str, Any]:
        return {
            **self.model_loader.status(),
            "loading": self.model_loading,
            "confidence_threshold": self.settings.confidence_threshold,
            "iou_threshold": self.settings.iou_threshold,
            "image_size": self.settings.image_size,
            "frame_skip": self.settings.frame_skip,
            "tracking": self.settings.enable_tracking,
            "masks": self.settings.enable_masks,
            "zone_alerts": self.settings.enable_zone_alerts,
            "detection_filter": self.detection_filter.status(self.model_loader.class_names),
        }

    def refresh_zones(self) -> None:
        self.zone_engine.set_zones(self.repository.list_zones())

    def detect_frame(self, frame: np.ndarray, confidence: float | None = None, iou: float | None = None) -> list[Detection]:
        if self.model_loader.model is None:
            return []
        detections = self.segmenter.infer(
            frame,
            confidence if confidence is not None else self.settings.confidence_threshold,
            iou if iou is not None else self.settings.iou_threshold,
            self.settings.image_size,
            self.settings.enable_tracking,
        )
        detections = self.detection_filter.apply(detections)
        return self.zone_engine.annotate_detections(detections)

    def _process_loop(self) -> None:
        while self.processing:
            frame = self.camera.read_latest()
            if frame is None:
                time.sleep(0.02)
                continue
            self._frame_index += 1
            if self._frame_index % self.settings.frame_skip != 0:
                continue
            try:
                detections: list[Detection] = []
                if self.model_loader.model is not None:
                    detections = self.segmenter.infer(
                        frame,
                        self.settings.confidence_threshold,
                        self.settings.iou_threshold,
                        self.settings.image_size,
                        self.settings.enable_tracking,
                    )
                detections = self.detection_filter.apply(detections)
                detections = self.zone_engine.annotate_detections(detections)
                occupancy = self.zone_engine.zone_occupancy(detections)
                fps = self.analytics.tick()
                summary = self.analytics.update(detections, occupancy).__dict__
                created_alerts = []
                if self.settings.enable_zone_alerts:
                    events = self.alerts.restricted_zone_events(self.zone_engine.restricted_violations(detections))
                    for event in events:
                        screenshot_path = self._save_alert_screenshot(frame, event.event_type)
                        event.screenshot_path = str(screenshot_path)
                        created_alerts.append(asdict(self.repository.create_event(event)))

                annotated = draw_annotations(frame, detections, self.zone_engine.zones)
                self.latest_frame_jpeg = encode_jpeg(annotated)
                self.latest_payload = {
                    "timestamp": time.time(),
                    "fps": fps,
                    "camera": self.camera.status(),
                    "model": self.model_status(),
                    "detections": [asdict(detection) for detection in detections],
                    "alerts": created_alerts,
                    "analytics": summary,
                }
                self.last_error = None
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.exception("Vision pipeline error")
                time.sleep(0.1)

    def _save_alert_screenshot(self, frame: np.ndarray, event_type: str) -> Path:
        filename = f"{event_type}-{int(time.time() * 1000)}.jpg"
        return save_jpeg(frame, self.settings.screenshot_dir / filename)

    def _filter_settings(self) -> DetectionFilterSettings:
        return DetectionFilterSettings(
            detection_mode=self.settings.detection_mode,  # type: ignore[arg-type]
            allowed_classes=self.settings.allowed_class_list,
            blocked_classes=self.settings.blocked_class_list,
        )
