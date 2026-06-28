"""Train the emotion classifier on Python 3.14 using MediaPipe **Tasks**.

This is the runs-on-this-machine counterpart to ``train_mdistortion_de`` (which
needs the unavailable ``mediapipe.solutions`` + a trained YOLO detector). Here the
Tasks ``FaceLandmarker`` does detection + landmarks in one step, so no YOLO model
is required to get an emotion classifier working.

Flow (same natural-computing core as the designed pipeline):

1. Landmark every emotion image with ``FaceLandmarker``.
2. **Differential Evolution** warps the node groups (per-group anisotropic
   scaling) for best class separability.
3. Build the three Mdistortion matrices and learn per-emotion **ranges**.
4. Train a classifier and save a ``joblib`` the live demo can load.

Example::

    affect-pi-train-emotion --max-per-class 120 --de-iterations 6 --de-popsize 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover - runtime dependency
    joblib = None

from .mdistortion_tasks import DEFAULT_MODEL, detect_image, face_from_pts, load_landmarker
from .train_mdistortion_de import (
    OFFSET_NAMES,
    build_dataset,
    compute_mdistortion_ranges,
    gather_emotion_samples,
    optimize_node_offsets,
    range_band_accuracy,
    ranges_summary,
    ranges_to_jsonable,
    train_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train emotion classifier via MediaPipe Tasks FaceLandmarker.")
    p.add_argument("--emotions-dir", type=Path, default=Path("data/emotions/Data"))
    p.add_argument("--landmarker-model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--max-per-class", type=int, default=120, help="Cap images per class (keep small for a quick model)")
    p.add_argument("--output-model", type=Path, default=Path("artifacts/emotion_tasks_model.joblib"))
    p.add_argument("--output-report", type=Path, default=Path("artifacts/emotion_tasks_report.json"))
    p.add_argument("--output-ranges", type=Path, default=Path("artifacts/emotion_tasks_ranges.json"))
    p.add_argument("--de-iterations", type=int, default=6, help="DE generations (0 or --no-de to skip)")
    p.add_argument("--de-popsize", type=int, default=12)
    p.add_argument("--de-f", type=float, default=0.6)
    p.add_argument("--de-cr", type=float, default=0.9)
    p.add_argument("--no-de", action="store_true", help="Skip DE; use zero node offsets")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def extract_faces(samples, landmarker):
    detected = []
    n = len(samples)
    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample.image_path))
        if img is None:
            continue
        pts = detect_image(landmarker, img)
        if pts is None:
            continue
        detected.append(face_from_pts(pts, label=sample.label, image_path=sample.image_path))
        if (i + 1) % 100 == 0:
            print(f"  landmarked {i + 1}/{n} images, {len(detected)} faces so far")
    return detected


def main() -> None:
    args = parse_args()
    if joblib is None:
        raise RuntimeError("joblib is required. Install with: pip install joblib")

    landmarker = load_landmarker(args.landmarker_model, video=False)

    samples = gather_emotion_samples(args.emotions_dir, args.max_per_class)
    if not samples:
        raise RuntimeError(f"No images found under {args.emotions_dir}")
    print(f"Landmarking {len(samples)} images with FaceLandmarker...")
    detected = extract_faces(samples, landmarker)
    landmarker.close()
    print(f"Usable faces: {len(detected)} / {len(samples)}")
    if len(detected) < 40:
        raise RuntimeError("Too few detectable faces. Increase --max-per-class.")

    if args.no_de or args.de_iterations <= 0:
        offsets = np.zeros(6, dtype=np.float32)
        node_opt = {"optimizer": "none", "offsets": offsets.tolist(), "offsets_named": dict(zip(OFFSET_NAMES, offsets.tolist()))}
    else:
        node_opt = optimize_node_offsets(
            detected_faces=detected,
            seed=args.seed,
            iters=args.de_iterations,
            popsize=args.de_popsize,
            f=args.de_f,
            cr=args.de_cr,
        )
        offsets = np.array(node_opt["offsets"], dtype=np.float32)

    features, labels, used_images = build_dataset(detected, offsets)
    model, acc, model_name = train_model(features, labels, seed=args.seed)

    ranges = compute_mdistortion_ranges(detected, offsets)
    ranges_json = ranges_to_jsonable(ranges)
    range_acc = range_band_accuracy(detected, offsets, ranges)

    classes = sorted(set(labels.tolist()))
    report = {
        "n_samples_input": len(samples),
        "n_samples_detected": len(detected),
        "n_samples_used": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "classes": classes,
        "landmarker_model": str(args.landmarker_model),
        "node_optimization": node_opt,
        "training": {"model": model_name, "accuracy": acc},
        "mdistortion_ranges": {
            "summary": ranges_summary(ranges),
            "range_band_accuracy_in_sample": range_acc,
            "saved": str(args.output_ranges),
        },
        "saved": {
            "model_weights": str(args.output_model),
            "report": str(args.output_report),
            "ranges": str(args.output_ranges),
        },
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": [f"f_{i}" for i in range(int(features.shape[1]))],
            "node_offsets": offsets.tolist(),
            "landmarker_model": str(args.landmarker_model),
            "classes": classes,
            "mdistortion_ranges": ranges_json,
        },
        args.output_model,
    )
    with args.output_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with args.output_ranges.open("w", encoding="utf-8") as f:
        json.dump(ranges_json, f, indent=2)

    print(f"\nSaved model: {args.output_model}")
    print(f"Saved report: {args.output_report}")
    print(f"Validation accuracy: {acc:.3f} ({model_name}) over classes {classes}")
    print("Run the live demo with classification:")
    print(f"  affect-pi-live --mirror --emotion-model {args.output_model}")


if __name__ == "__main__":
    main()
