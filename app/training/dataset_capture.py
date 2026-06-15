from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def capture_frames(source: str, output_dir: Path, interval_seconds: float, max_frames: int | None) -> int:
    capture_source: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    last_save = 0.0
    try:
        while max_frames is None or saved < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera read failed during dataset capture")
            now = time.time()
            if now - last_save >= interval_seconds:
                path = output_dir / f"frame_{int(now * 1000)}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Failed to write {path}")
                saved += 1
                last_save = now
                print(f"saved {path}")
            cv2.waitKey(1)
    finally:
        cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real camera frames for warehouse segmentation datasets.")
    parser.add_argument("--source", default="0", help="Webcam index, USB camera index, RTSP URL, or IP camera URL")
    parser.add_argument("--output", default="../../datasets/raw", help="Output directory for captured JPG frames")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between saved frames")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    capture_frames(args.source, Path(args.output).resolve(), args.interval, args.max_frames)


if __name__ == "__main__":
    main()

