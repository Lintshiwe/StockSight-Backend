from __future__ import annotations

import argparse
from pathlib import Path


def export(weights: Path, fmt: str, image_size: int) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.export(format=fmt, imgsz=image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO segmentation weights.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    export(Path(args.weights).resolve(), args.format, args.imgsz)


if __name__ == "__main__":
    main()

