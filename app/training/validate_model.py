from __future__ import annotations

import argparse
from pathlib import Path


def validate(weights: Path, data_yaml: Path, image_size: int) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.val(data=str(data_yaml), imgsz=image_size, task="segment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO segmentation weights.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="../../datasets/warehouse-seg/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    validate(Path(args.weights).resolve(), Path(args.data).resolve(), args.imgsz)


if __name__ == "__main__":
    main()

