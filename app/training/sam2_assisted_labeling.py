from __future__ import annotations

import argparse
from pathlib import Path


def run_assisted_labeling(input_dir: Path, output_dir: Path) -> None:
    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("SAM2-assisted labeling requires OpenCV, PyTorch, and a SAM2 installation") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    raise RuntimeError(
        "SAM2 is optional and not bundled. Install SAM2 and connect its predictor here for mask refinement; "
        "live inference uses YOLO segmentation by default."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional SAM2-assisted segmentation refinement hook.")
    parser.add_argument("--input", default="../../datasets/raw")
    parser.add_argument("--output", default="../../datasets/sam2-assisted")
    args = parser.parse_args()
    run_assisted_labeling(Path(args.input).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()

