from __future__ import annotations

import argparse
import json
import time
from collections import deque

import cv2

from .camera import make_camera
from .detectors import FaceMeshDetector, PoseDetector
from .model import make_model
from .pipeline import FrameResult, VisionPipeline
from .status import VisionStatus
from .trend import GaussianTrendLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face/body visibility and facial distortion pipeline")
    parser.add_argument("--camera", choices=["webcam", "picamera2"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Show debug video window")
    parser.add_argument("--print-every", type=int, default=15, help="Print every N frames")
    parser.add_argument("--trend-alpha", type=float, default=0.03)
    return parser.parse_args()


def result_summary(result: FrameResult) -> dict:
    base = {"status": result.status.value}
    if result.status == VisionStatus.FACE_VISIBLE:
        base.update(
            {
                "emotion": result.emotion.label if result.emotion else None,
                "probabilities": result.emotion.probabilities if result.emotion else {},
                "trend_count": result.trend.count if result.trend else 0,
                "trend_anomaly_score": result.trend.anomaly_score if result.trend else 0.0,
                "scalars": result.facial_distortions.scalars if result.facial_distortions else {},
                **result.meta,
            }
        )
    elif result.status == VisionStatus.BODY_VISIBLE:
        visible_names = sorted(result.body_parts.keys()) if result.body_parts else []
        base.update({"visible_body_parts": visible_names, **result.meta})
    return base


def draw_overlay(frame, result: FrameResult, fps: float) -> None:
    y = 30
    lines = [f"status={result.status.value}", f"fps={fps:.1f}"]
    if result.emotion:
        lines.append(f"emotion={result.emotion.label}")
    if result.trend:
        lines.append(f"trend_anomaly={result.trend.anomaly_score:.3f}")
    if result.body_parts:
        lines.append(f"body_parts={len(result.body_parts)}")
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 26


def main() -> None:
    args = parse_args()
    camera = make_camera(args.camera, args.camera_index, args.width, args.height)
    pipeline = VisionPipeline(
        face_detector=FaceMeshDetector(refine_landmarks=True),
        pose_detector=PoseDetector(),
        emotion_model=make_model(args.model_path),
        trend_layer=GaussianTrendLayer(alpha=args.trend_alpha),
    )

    frame_idx = 0
    times = deque(maxlen=30)
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print(json.dumps({"status": "CAMERA_READ_FAILED"}))
                break

            t0 = time.perf_counter()
            result = pipeline.process(frame)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            fps = 1.0 / (sum(times) / len(times)) if times else 0.0

            if frame_idx % args.print_every == 0:
                print(json.dumps(result_summary(result), default=float))

            if args.show:
                draw_overlay(frame, result, fps)
                cv2.imshow("affect-pi-base", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        pipeline.close()
        camera.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
