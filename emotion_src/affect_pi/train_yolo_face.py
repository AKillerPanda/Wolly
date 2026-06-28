"""Train the YOLO face detector used by the Mdistortion pipeline.

This is the missing first step of the requested flow: it turns the labelled face
boxes in ``faces.csv`` into a YOLO detection dataset and trains a detector, so
``affect-pi-train`` (train_mdistortion_de.py) has a ``best.pt`` to work with.

Two phases:

1. ``convert`` - group ``faces.csv`` by image, write one YOLO label file per
   image (``class cx cy w h`` normalized), split train/val at the image level,
   and emit a ``data.yaml``.
2. ``train``   - run ``ultralytics`` training on that dataset.

The labels are written next to the images as a ``labels`` sibling of the images
directory, which is the layout Ultralytics expects (it maps ``/images/`` to
``/labels/`` automatically), so no large image copy is needed.

Examples::

    # one shot: convert + train
    affect-pi-train-yolo --epochs 60 --imgsz 640

    # just build the dataset, do not train yet
    affect-pi-train-yolo --convert-only

    # train from an already-built dataset
    affect-pi-train-yolo --data-yaml runs/face_dataset/data.yaml --skip-convert
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - runtime dependency
    YOLO = None


FACE_CLASS_ID = 0
FACE_CLASS_NAME = "face"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert faces.csv to a YOLO dataset and train a face detector."
    )
    parser.add_argument(
        "--faces-csv",
        type=Path,
        default=Path("data/faces/faces.csv"),
        help="CSV with columns image_name,width,height,x0,y0,x1,y1",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/faces/images"),
        help="Directory holding the images referenced by faces.csv",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("runs/face_dataset"),
        help="Where to write train.txt/val.txt/data.yaml",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Where to write YOLO .txt labels (default: 'labels' sibling of --images-dir)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of images held out for validation",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n.pt",
        help="Base weights to fine-tune (e.g. yolo11n.pt, yolo26n.pt). Auto-downloads if absent.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (use -1 to let Ultralytics auto-pick on GPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device: 'cpu', '0', '0,1', etc.",
    )
    parser.add_argument("--project", type=str, default="runs/detect", help="Ultralytics project dir")
    parser.add_argument("--name", type=str, default="face_detector", help="Ultralytics run name")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for split + training")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Build the dataset and exit without training",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip dataset building and train from --data-yaml directly",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Existing data.yaml to train from when --skip-convert is set",
    )
    return parser.parse_args()


def read_boxes_by_image(
    csv_path: Path,
) -> dict[str, tuple[int, int, list[tuple[float, float, float, float]]]]:
    """Group faces.csv rows by image -> (width, height, [boxes])."""
    if not csv_path.exists():
        raise FileNotFoundError(f"faces.csv not found at {csv_path}")

    by_image: dict[str, tuple[int, int, list[tuple[float, float, float, float]]]] = {}
    grouped: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    dims: dict[str, tuple[int, int]] = {}

    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["image_name"]
            w, h = int(float(row["width"])), int(float(row["height"]))
            x0, y0, x1, y1 = (
                float(row["x0"]),
                float(row["y0"]),
                float(row["x1"]),
                float(row["y1"]),
            )
            if x1 <= x0 or y1 <= y0:
                continue  # drop degenerate boxes
            dims[name] = (w, h)
            grouped[name].append((x0, y0, x1, y1))

    for name, boxes in grouped.items():
        w, h = dims[name]
        by_image[name] = (w, h, boxes)
    return by_image


def box_to_yolo_line(box: tuple[float, float, float, float], w: int, h: int) -> str | None:
    """Convert an absolute-pixel xyxy box to a normalized YOLO label line."""
    x0, y0, x1, y1 = box
    # Clamp to the image to be safe, then normalize.
    x0 = min(max(x0, 0.0), w)
    x1 = min(max(x1, 0.0), w)
    y0 = min(max(y0, 0.0), h)
    y1 = min(max(y1, 0.0), h)
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    if bw <= 0.0 or bh <= 0.0:
        return None
    return f"{FACE_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def convert_dataset(args: argparse.Namespace) -> Path:
    import random

    images_dir: Path = args.images_dir
    labels_dir: Path = args.labels_dir or (images_dir.parent / "labels")
    dataset_dir: Path = args.dataset_dir
    labels_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    by_image = read_boxes_by_image(args.faces_csv)

    usable_images: list[Path] = []
    n_boxes = 0
    for name, (w, h, boxes) in sorted(by_image.items()):
        image_path = images_dir / name
        if not image_path.exists():
            continue

        lines = [line for box in boxes if (line := box_to_yolo_line(box, w, h)) is not None]
        if not lines:
            continue

        label_path = labels_dir / f"{Path(name).stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        usable_images.append(image_path.resolve())
        n_boxes += len(lines)

    if not usable_images:
        raise RuntimeError("No usable (image, label) pairs were produced. Check paths in faces.csv.")

    rng = random.Random(args.seed)
    rng.shuffle(usable_images)
    n_val = max(1, int(len(usable_images) * args.val_split))
    val_images = usable_images[:n_val]
    train_images = usable_images[n_val:]

    train_txt = dataset_dir / "train.txt"
    val_txt = dataset_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_images) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(str(p) for p in val_images) + "\n", encoding="utf-8")

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "# Auto-generated by train_yolo_face.py\n"
        f"path: {dataset_dir.resolve()}\n"
        f"train: {train_txt.resolve()}\n"
        f"val: {val_txt.resolve()}\n"
        "names:\n"
        f"  {FACE_CLASS_ID}: {FACE_CLASS_NAME}\n",
        encoding="utf-8",
    )

    print(
        f"Dataset ready: {len(train_images)} train + {len(val_images)} val images, "
        f"{n_boxes} boxes.\n"
        f"  labels:   {labels_dir.resolve()}\n"
        f"  data.yaml: {data_yaml.resolve()}"
    )
    return data_yaml


def train_detector(args: argparse.Namespace, data_yaml: Path) -> None:
    if YOLO is None:
        raise RuntimeError("ultralytics is required. Install with: pip install ultralytics")

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=True,
    )

    best = getattr(getattr(model, "trainer", None), "best", None)
    save_dir = getattr(results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    print("\nTraining complete.")
    if best:
        print(f"Best weights: {best}")
        print("Use it with:")
        print(f"  affect-pi-train --yolo-face-model {best} --tune-yolo")
    elif save_dir:
        print(f"Run directory: {save_dir} (best weights under weights/best.pt)")


def main() -> None:
    args = parse_args()

    if args.skip_convert:
        data_yaml = args.data_yaml
        if data_yaml is None or not data_yaml.exists():
            raise RuntimeError("--skip-convert requires an existing --data-yaml")
    else:
        data_yaml = convert_dataset(args)

    if args.convert_only:
        print("Conversion done (--convert-only set); skipping training.")
        return

    train_detector(args, data_yaml)


if __name__ == "__main__":
    main()
