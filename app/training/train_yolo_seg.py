from __future__ import annotations

import argparse
from pathlib import Path


def train(data_yaml: Path, model: str, epochs: int, image_size: int, batch: int, resume: bool) -> None:
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        task="segment",
        project="runs",
        name="warehouse-seg",
        resume=resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or resume a YOLO segmentation model for warehouse classes.")
    parser.add_argument("--data", default="../../datasets/warehouse-seg/data.yaml")
    parser.add_argument("--model", default="../../models/yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(Path(args.data).resolve(), args.model, args.epochs, args.imgsz, args.batch, args.resume)


if __name__ == "__main__":
    main()

