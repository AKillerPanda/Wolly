"""Evaluate an emotion model on a held-out set and print a confusion matrix.

The test images are taken from a slice that is **disjoint** from the images used
for training (``--skip-per-class`` must be >= the trainer's ``--max-per-class``),
so the numbers reflect generalization, not memorization. It evaluates the exact
saved model + node offsets via the shared ``feature_from_pts``.

Example::

    affect-pi-eval-emotion --skip-per-class 120 --test-per-class 250
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import joblib
from sklearn.metrics import classification_report, confusion_matrix

from .mdistortion_tasks import DEFAULT_MODEL, detect_image, feature_from_pts, load_landmarker
from .train_mdistortion_de import FaceSample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confusion-matrix evaluation for an affect-pi emotion model.")
    p.add_argument("--emotion-model", type=Path, default=Path("artifacts/emotion_tasks_model.joblib"))
    p.add_argument("--emotions-dir", type=Path, default=Path("data/emotions/Data"))
    p.add_argument("--landmarker-model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--skip-per-class", type=int, default=120,
                   help="Skip this many images per class (must be >= training --max-per-class)")
    p.add_argument("--test-per-class", type=int, default=250, help="Test images per class after the skip")
    p.add_argument("--output-cm", type=Path, default=Path("artifacts/confusion_matrix.png"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/eval_report.json"))
    return p.parse_args()


def gather_test_samples(root: Path, skip: int, take: int) -> list[FaceSample]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    samples: list[FaceSample] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in exts]
        for image_path in images[skip:skip + take]:
            samples.append(FaceSample(image_path=image_path, label=class_dir.name))
    return samples


def print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    short = [lab[:7] for lab in labels]
    col_w = max(7, max(len(s) for s in short)) + 1
    head = " " * 12 + "".join(f"{s:>{col_w}}" for s in short)
    print("\nConfusion matrix  (rows = true, cols = predicted)")
    print(head)
    for i, lab in enumerate(labels):
        row = "".join(f"{int(cm[i, j]):>{col_w}}" for j in range(len(labels)))
        print(f"{lab[:11]:<12}{row}")
    # Row-normalized recall view.
    print("\nRow-normalized (recall %, rows = true)")
    print(head)
    for i, lab in enumerate(labels):
        total = cm[i].sum()
        row = "".join(f"{(100.0 * cm[i, j] / total if total else 0):>{col_w}.0f}" for j in range(len(labels)))
        print(f"{lab[:11]:<12}{row}")


def save_cm_png(cm: np.ndarray, labels: list[str], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    cmn = cm.astype(float)
    row_sums = cmn.sum(axis=1, keepdims=True)
    cmn = np.divide(cmn, row_sums, out=np.zeros_like(cmn), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Emotion confusion matrix (row-normalized)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j] * 100:.0f}%",
                    ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    payload = joblib.load(args.emotion_model)
    model = payload["model"]
    offsets = np.array(payload.get("node_offsets", [0.0] * 6), dtype=np.float32)
    landmarker = load_landmarker(args.landmarker_model, video=False)

    samples = gather_test_samples(args.emotions_dir, args.skip_per_class, args.test_per_class)
    if not samples:
        raise RuntimeError("No test samples found. Lower --skip-per-class or check --emotions-dir.")
    print(f"Evaluating on {len(samples)} held-out images (skip={args.skip_per_class}, take={args.test_per_class})...")

    y_true: list[str] = []
    y_pred: list[str] = []
    no_face = 0
    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample.image_path))
        if img is None:
            continue
        pts = detect_image(landmarker, img)
        if pts is None:
            no_face += 1
            continue
        feat = feature_from_pts(pts, offsets)
        if feat is None:
            no_face += 1
            continue
        pred = model.predict(feat.reshape(1, -1))[0]
        y_true.append(sample.label)
        y_pred.append(str(pred))
        if (i + 1) % 200 == 0:
            print(f"  evaluated {i + 1}/{len(samples)}")
    landmarker.close()

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

    print_confusion_matrix(cm, labels)
    print(f"\nOverall accuracy: {acc:.3f} on {len(y_true)} faces "
          f"({no_face} images had no detectable face)")
    report = classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0)
    print("\nPer-class report:")
    print(report)

    png_ok = save_cm_png(cm, labels, args.output_cm)
    print(("Saved confusion matrix image: " + str(args.output_cm)) if png_ok
          else "matplotlib not available; skipped PNG.")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump({
            "n_test_faces": len(y_true),
            "no_face_images": no_face,
            "accuracy": acc,
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        }, f, indent=2)
    print(f"Saved eval JSON: {args.output_json}")


if __name__ == "__main__":
    main()
