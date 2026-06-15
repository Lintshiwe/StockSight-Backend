from app.storage.models import Detection
from app.vision.detection_filter import DetectionFilter, DetectionFilterSettings


def test_retail_fallback_blocks_person_and_cars() -> None:
    detections = [
        Detection("person", 0.99, [0, 0, 10, 10], centroid=(5, 5)),
        Detection("car", 0.91, [10, 0, 20, 10], centroid=(15, 5)),
        Detection("couch", 0.85, [20, 0, 40, 20], centroid=(30, 10)),
        Detection("refrigerator", 0.87, [40, 0, 70, 30], centroid=(55, 15)),
    ]

    filtered = DetectionFilter(DetectionFilterSettings(detection_mode="retail_coco_fallback")).apply(detections)

    assert [d.class_name for d in filtered] == ["couch", "refrigerator"]


def test_strict_warehouse_retail_allows_custom_product_classes_only() -> None:
    detections = [
        Detection("person", 0.99, [0, 0, 10, 10], centroid=(5, 5)),
        Detection("couch", 0.85, [20, 0, 40, 20], centroid=(30, 10)),
        Detection("package", 0.9, [40, 0, 70, 30], centroid=(55, 15)),
        Detection("iron", 0.9, [80, 0, 100, 20], centroid=(90, 10)),
    ]

    filtered = DetectionFilter(DetectionFilterSettings(detection_mode="strict_warehouse_retail")).apply(detections)

    assert [d.class_name for d in filtered] == ["package", "iron"]

