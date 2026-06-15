from __future__ import annotations

from typing import Any

import numpy as np

from app.storage.models import Detection
from app.vision.mask_utils import bbox_centroid, polygon_area
from app.vision.tracker import CentroidTracker


class Segmenter:
    def __init__(self, model_loader: Any, tracker: CentroidTracker | None = None) -> None:
        self.model_loader = model_loader
        self.tracker = tracker or CentroidTracker()

    def infer(
        self,
        frame: np.ndarray,
        confidence: float,
        iou: float,
        image_size: int,
        enable_tracking: bool = True,
    ) -> list[Detection]:
        if self.model_loader.model is None:
            raise RuntimeError("No YOLO segmentation model is loaded")

        if enable_tracking:
            results = self.model_loader.model.track(
                frame,
                persist=True,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                device=self.model_loader.device,
                verbose=False,
            )
        else:
            results = self.model_loader.model.predict(
                frame,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                device=self.model_loader.device,
                verbose=False,
            )
        detections = self._parse_results(results)
        if enable_tracking:
            self.tracker.update(detections)
        return detections

    def _parse_results(self, results: Any) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else [None] * len(xyxy)
        masks = self._extract_masks(result)

        for idx, bbox_arr in enumerate(xyxy):
            bbox = [float(v) for v in bbox_arr.tolist()]
            class_id = int(classes[idx])
            class_name = self.model_loader.class_names.get(class_id, str(class_id))
            mask = masks[idx] if idx < len(masks) else None
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(confs[idx]),
                    bbox=bbox,
                    mask=mask,
                    area=polygon_area(mask) or float(max(0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))),
                    track_id=None if ids[idx] is None else int(ids[idx]),
                    centroid=bbox_centroid(bbox),
                )
            )
        return detections

    def _extract_masks(self, result: Any) -> list[list[tuple[float, float]] | None]:
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None or getattr(masks_obj, "xy", None) is None:
            return []
        masks: list[list[tuple[float, float]] | None] = []
        for polygon in masks_obj.xy:
            if polygon is None:
                masks.append(None)
            else:
                masks.append([(float(x), float(y)) for x, y in polygon])
        return masks

