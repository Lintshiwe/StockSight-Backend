from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.storage.models import Detection


DetectionMode = Literal["strict_warehouse_retail", "retail_coco_fallback", "safety"]

STRICT_WAREHOUSE_RETAIL_CLASSES = {
    "box",
    "package",
    "pallet",
    "crate",
    "forklift",
    "shelf",
    "rack",
    "damaged_package",
    "spill",
    "appliance",
    "tissue",
    "iron",
    "sofa",
    "retail_item",
    "carton",
}

RETAIL_COCO_FALLBACK_CLASSES = {
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "bowl",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "mouse",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "toothbrush",
}

DEFAULT_BLOCKED_CLASSES = {
    "person",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bicycle",
}


@dataclass
class DetectionFilterSettings:
    detection_mode: DetectionMode = "retail_coco_fallback"
    allowed_classes: list[str] = field(default_factory=list)
    blocked_classes: list[str] = field(default_factory=lambda: sorted(DEFAULT_BLOCKED_CLASSES))


class DetectionFilter:
    def __init__(self, settings: DetectionFilterSettings | None = None) -> None:
        self.settings = settings or DetectionFilterSettings()

    def apply(self, detections: list[Detection]) -> list[Detection]:
        allowed = self.allowed_class_set()
        blocked = set(self.settings.blocked_classes)
        return [
            detection
            for detection in detections
            if detection.class_name not in blocked and detection.class_name in allowed
        ]

    def allowed_class_set(self) -> set[str]:
        if self.settings.allowed_classes:
            return set(self.settings.allowed_classes)
        if self.settings.detection_mode == "strict_warehouse_retail":
            return set(STRICT_WAREHOUSE_RETAIL_CLASSES)
        if self.settings.detection_mode == "retail_coco_fallback":
            return set(STRICT_WAREHOUSE_RETAIL_CLASSES) | set(RETAIL_COCO_FALLBACK_CLASSES)
        return set(STRICT_WAREHOUSE_RETAIL_CLASSES) | {"person", "helmet", "safety_vest", "forklift"}

    def status(self, loaded_class_names: dict[int, str] | dict[str, str] | None = None) -> dict[str, object]:
        loaded = set((loaded_class_names or {}).values())
        allowed = self.allowed_class_set()
        matching = sorted(loaded & allowed)
        blocked_loaded = sorted(loaded & set(self.settings.blocked_classes))
        return {
            "detection_mode": self.settings.detection_mode,
            "allowed_classes": sorted(allowed),
            "blocked_classes": sorted(self.settings.blocked_classes),
            "matching_loaded_classes": matching,
            "blocked_loaded_classes": blocked_loaded,
            "filter_warning": None
            if matching
            else "Loaded model has no classes matching the active retail/warehouse detection mode.",
        }
