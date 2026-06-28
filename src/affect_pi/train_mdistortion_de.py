from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .DE_nodes import de_optimize
from .DFO_image import dfo_optimize

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - runtime dependency
    YOLO = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - runtime dependency
    mp = None

try:
    import joblib
except Exception:  # pragma: no cover - runtime dependency
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - optional fallback
    RandomForestClassifier = None
    train_test_split = None


LEFT_EYE_SKIN_IDS = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7,
]

RIGHT_EYE_SKIN_IDS = [
    263, 466, 388, 387, 386, 385, 384, 398,
    362, 382, 381, 380, 374, 373, 390, 249,
]

OUTER_MOUTH_IDS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]


@dataclass
class FaceSample:
    image_path: Path
    label: str


@dataclass
class DetectedFace:
    image_path: Path
    label: str
    landmarks: dict[int, np.ndarray]
    face_width: float
    face_height: float


class NearestCentroidModel:
    def __init__(self, centroids: dict[str, np.ndarray]):
        self.centroids = centroids
        self.classes_ = np.array(sorted(centroids.keys()))
        # Stack centroids once (n_classes, dim) so scoring is a single broadcast.
        self._matrix = np.stack([np.asarray(centroids[c], dtype=np.float32)
                                 for c in self.classes_], axis=0)

    def _distances(self, x: np.ndarray) -> np.ndarray:
        """(n_samples, n_classes) distance matrix, fully vectorized."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        diff = x[:, None, :] - self._matrix[None, :, :]
        return np.sqrt(np.einsum("ncd,ncd->nc", diff, diff))

    def predict(self, x: np.ndarray):
        idx = np.argmin(self._distances(x), axis=1)
        return self.classes_[idx]

    def predict_proba(self, x: np.ndarray):
        inv = 1.0 / (self._distances(x) + 1e-8)   # distances -> pseudo-probabilities
        return (inv / inv.sum(axis=1, keepdims=True)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an emotion classifier using YOLO face detection + MediaPipe nodes + "
            "DE optimization over three Mdistortion matrices (mouth/left eye/right eye)."
        )
    )
    parser.add_argument(
        "--yolo-face-model",
        required=True,
        help="Path to YOLO face detector weights (best.pt)",
    )
    parser.add_argument(
        "--emotions-dir",
        type=Path,
        default=Path("data/emotions/Data"),
        help="Root directory with class folders of emotion face images",
    )
    parser.add_argument(
        "--faces-csv",
        type=Path,
        default=Path("data/faces/faces.csv"),
        help="Optional face bbox CSV used for YOLO hyperparameter tuning",
    )
    parser.add_argument(
        "--faces-images-dir",
        type=Path,
        default=Path("data/faces/images"),
        help="Image directory for faces.csv",
    )
    parser.add_argument(
        "--output-model",
        "--save-weights",
        dest="output_model",
        type=Path,
        default=Path("artifacts/mdistortion_emotion_model.joblib"),
        help="Path to save trained model weights/artifact (.joblib)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("artifacts/mdistortion_training_report.json"),
        help="Output report JSON path",
    )
    parser.add_argument(
        "--output-ranges",
        type=Path,
        default=Path("artifacts/mdistortion_ranges.json"),
        help="Per-emotion Mdistortion range table JSON path",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=300,
        help="Cap the number of images per class for training",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.30,
        help="Initial YOLO confidence",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Initial YOLO inference image size",
    )
    parser.add_argument(
        "--roi-pad",
        type=float,
        default=0.15,
        help="ROI padding applied before MediaPipe",
    )
    parser.add_argument(
        "--tune-yolo",
        action="store_true",
        help="Run DFO optimization for YOLO conf/imgsz/roi-pad using faces.csv IoU objective",
    )
    parser.add_argument(
        "--dfo-iterations",
        type=int,
        default=40,
        help="DFO iterations for YOLO hyperparameter tuning",
    )
    parser.add_argument(
        "--de-iterations",
        type=int,
        default=40,
        help="Differential Evolution generations for node-offset optimization",
    )
    parser.add_argument(
        "--de-popsize",
        type=int,
        default=30,
        help="Differential Evolution population size for node-offset optimization",
    )
    parser.add_argument(
        "--de-f",
        type=float,
        default=0.6,
        help="Differential Evolution differential weight F (mutation factor)",
    )
    parser.add_argument(
        "--de-cr",
        type=float,
        default=0.9,
        help="Differential Evolution crossover probability CR",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    return parser.parse_args()


def require_dependencies() -> None:
    if YOLO is None:
        raise RuntimeError("ultralytics is required. Install with: pip install ultralytics")
    if mp is None:
        raise RuntimeError("mediapipe is required. Install with: pip install mediapipe")
    if joblib is None:
        raise RuntimeError("joblib is required. Install with: pip install joblib")


def gather_emotion_samples(root: Path, max_per_class: int) -> list[FaceSample]:
    samples: list[FaceSample] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in exts]
        for image_path in images[:max_per_class]:
            samples.append(FaceSample(image_path=image_path, label=class_dir.name))
    return samples


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-8 else 0.0


def read_faces_csv_rows(csv_path: Path, images_dir: Path, limit: int = 250) -> list[tuple[Path, tuple[float, float, float, float]]]:
    rows: list[tuple[Path, tuple[float, float, float, float]]] = []
    if not csv_path.exists():
        return rows

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image_name"]
            image_path = images_dir / image_name
            if not image_path.exists():
                continue
            gt = (float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"]))
            rows.append((image_path, gt))
            if len(rows) >= limit:
                break
    return rows


def predict_primary_box(model, image: np.ndarray, conf: float, imgsz: int):
    results = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return None
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones((len(xyxy),))
    idx = int(np.argmax(scores))
    x1, y1, x2, y2 = xyxy[idx].astype(float).tolist()
    return x1, y1, x2, y2


def tune_yolo_hparams(model, faces_rows, seed: int, iters: int):
    if not faces_rows:
        return None

    def objective(x: np.ndarray) -> float:
        conf = float(np.clip(x[0], 0.05, 0.85))
        imgsz = int(np.clip(round(x[1]), 320, 1024))

        ious: list[float] = []
        for image_path, gt in faces_rows:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            pred = predict_primary_box(model, img, conf=conf, imgsz=imgsz)
            if pred is None:
                ious.append(0.0)
            else:
                ious.append(iou_xyxy(pred, gt))

        if not ious:
            return 1.0
        return 1.0 - float(np.mean(ious))

    result = dfo_optimize(
        objective=objective,
        bounds=[(0.05, 0.85), (320.0, 1024.0)],
        population_size=25,
        max_iterations=iters,
        seed=seed,
    )
    return {
        "best_conf": float(np.clip(result.best_position[0], 0.05, 0.85)),
        "best_imgsz": int(np.clip(round(float(result.best_position[1])), 320, 1024)),
        "best_loss": float(result.best_fitness),
    }


def extract_landmarks_in_roi(face_mesh, frame_bgr: np.ndarray, box, roi_pad: float):
    x1, y1, x2, y2 = box
    h, w = frame_bgr.shape[:2]
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    px = int(bw * roi_pad)
    py = int(bh * roi_pad)

    rx1 = max(0, int(x1) - px)
    ry1 = max(0, int(y1) - py)
    rx2 = min(w - 1, int(x2) + px)
    ry2 = min(h - 1, int(y2) + py)
    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(roi_rgb)
    if not result.multi_face_landmarks:
        return None

    landmarks = {}
    for i, lm in enumerate(result.multi_face_landmarks[0].landmark):
        landmarks[i] = np.array([rx1 + lm.x * roi_w, ry1 + lm.y * roi_h], dtype=np.float32)

    return landmarks, float(bw), float(bh)


def gather_detected_faces(samples, yolo_model, conf: float, imgsz: int, roi_pad: float):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    detected: list[DetectedFace] = []
    for sample in samples:
        img = cv2.imread(str(sample.image_path))
        if img is None:
            continue

        box = predict_primary_box(yolo_model, img, conf=conf, imgsz=imgsz)
        if box is None:
            continue

        lm_data = extract_landmarks_in_roi(face_mesh, img, box, roi_pad=roi_pad)
        if lm_data is None:
            continue

        landmarks, bw, bh = lm_data
        detected.append(
            DetectedFace(
                image_path=sample.image_path,
                label=sample.label,
                landmarks=landmarks,
                face_width=bw,
                face_height=bh,
            )
        )

    face_mesh.close()
    return detected


def warp_group(landmarks: dict[int, np.ndarray], ids: list[int], sx: float, sy: float):
    """Anisotropically scale a node group about its own centroid.

    DE searches these per-group ``(sx, sy)`` factors. Scaling x and y by
    *different* amounts changes the group's normalized pairwise-distance matrix,
    which is what makes the optimization meaningful: a rigid translation (the old
    behaviour) left every intra-group distance unchanged, so the feature -- and
    therefore DE's separability objective -- was completely insensitive to it.
    A uniform scale would also cancel under the max-normalization in
    ``pairwise_mdistortion``; only the *anisotropy* (sx != sy) survives, so DE is
    effectively choosing how much to emphasise horizontal vs vertical spread per
    region to best separate the emotion classes.
    """
    pts = [landmarks[idx] for idx in ids if idx in landmarks]
    if len(pts) < 2:
        return None
    arr = np.stack(pts, axis=0).astype(np.float32)
    centroid = arr.mean(axis=0)
    scale = np.array([1.0 + sx, 1.0 + sy], dtype=np.float32)
    return (centroid + (arr - centroid) * scale).astype(np.float32)


def pairwise_mdistortion(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    d = np.linalg.norm(diff, axis=-1).astype(np.float32)
    max_d = float(np.max(d))
    if max_d > 1e-6:
        d = d / max_d
    return d


def sample_feature(face: DetectedFace, offsets: np.ndarray):
    mx, my, lx, ly, rx, ry = [float(v) for v in offsets]

    mouth = warp_group(face.landmarks, OUTER_MOUTH_IDS, mx, my)
    left_eye = warp_group(face.landmarks, LEFT_EYE_SKIN_IDS, lx, ly)
    right_eye = warp_group(face.landmarks, RIGHT_EYE_SKIN_IDS, rx, ry)

    if mouth is None or left_eye is None or right_eye is None:
        return None

    m1 = pairwise_mdistortion(mouth).ravel()
    m2 = pairwise_mdistortion(left_eye).ravel()
    m3 = pairwise_mdistortion(right_eye).ravel()
    return np.concatenate([m1, m2, m3], dtype=np.float32)


def fisher_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Between-class / within-class scatter ratio, fully vectorized (no per-class
    Python loop): class means via a single scatter-add, then both scatters as one
    array reduction each."""
    labels = np.asarray(labels)
    classes, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        return 0.0
    feats = features.astype(np.float64)
    means = np.zeros((len(classes), feats.shape[1]), dtype=np.float64)
    np.add.at(means, inv, feats)                 # sum per class
    means /= counts[:, None]                      # -> class means
    mu = feats.mean(axis=0)
    sb = float(np.sum(counts[:, None] * (means - mu) ** 2))      # between-class
    sw = float(np.sum((feats - means[inv]) ** 2))               # within-class
    return sb / (sw + 1e-8)


# Per-group anisotropic scale factors (sx, sy) DE optimizes; 0 = no scaling.
OFFSET_NAMES = ["mouth_sx", "mouth_sy", "left_eye_sx", "left_eye_sy", "right_eye_sx", "right_eye_sy"]


def optimize_node_offsets(
    detected_faces: list[DetectedFace],
    seed: int,
    iters: int,
    popsize: int = 30,
    f: float = 0.6,
    cr: float = 0.9,
):
    # Gather the per-group point tensors ONCE; each DE candidate then only does
    # vectorized warp + pairwise + Fisher over the whole batch (no per-face loop).
    tensors, labels, kept = gather_group_tensors(detected_faces)

    def objective(x: np.ndarray) -> float:
        if len(kept) < 20:
            return 1e3
        feats = features_from_tensors(tensors, x)
        # Minimize negative separability => maximize separability.
        return -fisher_score(feats, labels)

    # Differential Evolution searches per-group anisotropic scale factors (sx, sy)
    # that warp each node group so the three Mdistortion matrices best separate the
    # emotion classes. Bounds give scales in [0.6, 1.4]; the objective genuinely
    # responds to these (unlike the old translation offsets, which were a no-op).
    result = de_optimize(
        objective=objective,
        bounds=[(-0.40, 0.40)] * 6,
        population_size=popsize,
        max_iterations=iters,
        f=f,
        cr=cr,
        seed=seed,
    )

    offsets = [float(v) for v in result.best_position.tolist()]
    return {
        "optimizer": "differential_evolution",
        "offsets": offsets,
        "offsets_named": dict(zip(OFFSET_NAMES, offsets)),
        "objective": float(result.best_fitness),
        "fisher_score": float(-result.best_fitness),
        "history_tail": result.history[-10:],
    }


def build_dataset(detected_faces: list[DetectedFace], offsets: np.ndarray):
    features, labels, kept = build_feature_matrix(detected_faces, offsets)
    if features.shape[0] == 0:
        raise RuntimeError("No usable features were generated. Check detections and offsets.")
    image_paths = [str(f.image_path) for f in kept]
    return features, labels, image_paths


# Three Mdistortion matrix types, one set per detected face.
MDISTORTION_GROUPS: dict[str, list[int]] = {
    "mouth": OUTER_MOUTH_IDS,
    "left_eye": LEFT_EYE_SKIN_IDS,
    "right_eye": RIGHT_EYE_SKIN_IDS,
}


def _group_offset(offsets: np.ndarray, group: str) -> tuple[float, float]:
    mx, my, lx, ly, rx, ry = [float(v) for v in offsets]
    return {"mouth": (mx, my), "left_eye": (lx, ly), "right_eye": (rx, ry)}[group]


# --------------------------------------------------------------------------- #
#  Batched (vectorized) feature math                                          #
#                                                                             #
#  Landmarks arrive as per-face dicts, so the only Python loop left is the    #
#  one-time gather into (N, K, 2) tensors. After that, warping, pairwise       #
#  distances, normalization and the upper triangle are all single array ops    #
#  over the whole dataset -- the per-face loops that used to wrap every DE     #
#  objective evaluation are gone. Outputs are identical to sample_feature /     #
#  group_mdistortion_upper (equivalence-tested).                               #
# --------------------------------------------------------------------------- #
_REQUIRED_IDS = set(OUTER_MOUTH_IDS) | set(LEFT_EYE_SKIN_IDS) | set(RIGHT_EYE_SKIN_IDS)


def _stack_group(faces: list[DetectedFace], ids: list[int]) -> np.ndarray:
    """(N, K, 2) array of one group's landmark points across faces."""
    out = np.empty((len(faces), len(ids), 2), dtype=np.float32)
    for n, face in enumerate(faces):
        for k, i in enumerate(ids):
            out[n, k] = face.landmarks[i]
    return out


def gather_group_tensors(detected_faces: list[DetectedFace]):
    """Keep faces that have every required landmark; return per-group (N, K, 2)
    tensors, the label array, and the kept faces (for image paths)."""
    kept = [f for f in detected_faces if _REQUIRED_IDS.issubset(f.landmarks.keys())]
    tensors = {g: _stack_group(kept, ids) for g, ids in MDISTORTION_GROUPS.items()}
    labels = np.array([f.label for f in kept])
    return tensors, labels, kept


def _warp_batch(pts: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Anisotropic scale about each face's group centroid. pts: (N, K, 2)."""
    centroid = pts.mean(axis=1, keepdims=True)                      # (N, 1, 2)
    scale = np.array([1.0 + sx, 1.0 + sy], dtype=np.float32)
    return centroid + (pts - centroid) * scale


def _pairwise_norm_batch(pts: np.ndarray) -> np.ndarray:
    """(N, K, K) per-face max-normalized pairwise-distance matrices for (N, K, 2)."""
    diff = pts[:, :, None, :] - pts[:, None, :, :]                  # (N, K, K, 2)
    d = np.sqrt(np.einsum("nijc,nijc->nij", diff, diff)).astype(np.float32)
    n = d.shape[0]
    maxd = d.reshape(n, -1).max(axis=1)
    denom = np.where(maxd > 1e-6, maxd, 1.0).astype(np.float32)
    return d / denom[:, None, None]


def features_from_tensors(tensors: dict[str, np.ndarray], offsets: np.ndarray) -> np.ndarray:
    """The (N, 912) Mdistortion feature matrix from pre-gathered group tensors."""
    parts = []
    for group, ids in MDISTORTION_GROUPS.items():
        sx, sy = _group_offset(offsets, group)
        warped = _warp_batch(tensors[group], sx, sy)
        parts.append(_pairwise_norm_batch(warped).reshape(warped.shape[0], -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _group_upper_batch(tensors: dict[str, np.ndarray], group: str, offsets: np.ndarray) -> np.ndarray:
    """(N, P) upper-triangle Mdistortion vectors for one group across all faces."""
    sx, sy = _group_offset(offsets, group)
    m = _pairwise_norm_batch(_warp_batch(tensors[group], sx, sy))   # (N, K, K)
    iu = np.triu_indices(m.shape[1], k=1)
    return m[:, iu[0], iu[1]]


def build_feature_matrix(detected_faces: list[DetectedFace], offsets: np.ndarray):
    """Vectorized counterpart of stacking sample_feature over faces."""
    tensors, labels, kept = gather_group_tensors(detected_faces)
    if not kept:
        return np.empty((0, 0), dtype=np.float32), labels, kept
    return features_from_tensors(tensors, offsets), labels, kept


def group_mdistortion_upper(face: DetectedFace, group: str, offsets: np.ndarray):
    """Upper-triangle Mdistortion vector for one group (every node-pair
    combination), or None if the group's landmarks are incomplete."""
    ids = MDISTORTION_GROUPS[group]
    sx, sy = _group_offset(offsets, group)
    pts = warp_group(face.landmarks, ids, sx, sy)
    if pts is None or pts.shape[0] != len(ids):
        return None
    m = pairwise_mdistortion(pts)
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def group_pair_labels(group: str) -> list[tuple[int, int]]:
    """The (landmark_a, landmark_b) identity of every combination in a group."""
    ids = MDISTORTION_GROUPS[group]
    iu = np.triu_indices(len(ids), k=1)
    return [(int(ids[i]), int(ids[j])) for i, j in zip(iu[0].tolist(), iu[1].tolist())]


def compute_mdistortion_ranges(detected_faces: list[DetectedFace], offsets: np.ndarray) -> dict:
    """Per-emotion min/max/mean/std for every node-pair combination in each of
    the three Mdistortion matrices (mouth / left eye / right eye)."""
    tensors, labels, kept = gather_group_tensors(detected_faces)
    ranges: dict[str, dict] = {}
    for group in MDISTORTION_GROUPS:
        ranges[group] = {"pairs": group_pair_labels(group), "by_emotion": {}}
        if not kept:
            continue
        vecs = _group_upper_batch(tensors, group, offsets)      # (N, P)
        for label in np.unique(labels):
            arr = vecs[labels == label]                          # (n_label, P)
            ranges[group]["by_emotion"][str(label)] = {
                "count": int(arr.shape[0]),
                "min": arr.min(axis=0),
                "max": arr.max(axis=0),
                "mean": arr.mean(axis=0),
                "std": arr.std(axis=0),
            }
    return ranges


def range_band_accuracy(detected_faces: list[DetectedFace], offsets: np.ndarray, ranges: dict) -> float:
    """In-sample check that the learned ranges are discriminative: predict the
    emotion whose [min, max] bands contain the most of a sample's pair values."""
    classes = sorted({lab for group in ranges for lab in ranges[group]["by_emotion"]})
    if len(classes) < 2:
        return 0.0

    tensors, labels, kept = gather_group_tensors(detected_faces)
    if not kept:
        return 0.0

    group_vecs = {g: _group_upper_batch(tensors, g, offsets) for g in MDISTORTION_GROUPS}  # (N, P) each
    n = len(kept)
    # For each class, the fraction of a sample's pair values that fall inside that
    # class's learned [min, max] bands -- computed for all faces at once.
    scores = np.zeros((n, len(classes)), dtype=np.float64)
    for ci, cls in enumerate(classes):
        in_band = np.zeros(n, dtype=np.float64)
        n_pairs = 0
        for group in MDISTORTION_GROUPS:
            band = ranges[group]["by_emotion"].get(cls)
            if band is None:
                continue
            v = group_vecs[group]
            in_band += ((v >= band["min"]) & (v <= band["max"])).sum(axis=1)
            n_pairs += v.shape[1]
        scores[:, ci] = in_band / n_pairs if n_pairs else 0.0

    pred = np.array(classes)[scores.argmax(axis=1)]    # argmax ties -> first class (matches old >)
    return float(np.mean(pred == labels))


def ranges_to_jsonable(ranges: dict) -> dict:
    """Full numeric range table with numpy arrays converted to lists."""
    out: dict = {}
    for group, payload in ranges.items():
        out[group] = {
            "pairs": [list(p) for p in payload["pairs"]],
            "by_emotion": {
                label: {
                    "count": band["count"],
                    "min": band["min"].tolist(),
                    "max": band["max"].tolist(),
                    "mean": band["mean"].tolist(),
                    "std": band["std"].tolist(),
                }
                for label, band in payload["by_emotion"].items()
            },
        }
    return out


def ranges_summary(ranges: dict) -> dict:
    """Compact, human-readable digest of the range table for the report."""
    summary: dict = {}
    for group, payload in ranges.items():
        summary[group] = {
            "n_pairs": len(payload["pairs"]),
            "emotions": {
                label: {
                    "count": band["count"],
                    "global_min": float(band["min"].min()),
                    "global_max": float(band["max"].max()),
                    "mean_of_means": float(band["mean"].mean()),
                }
                for label, band in payload["by_emotion"].items()
            },
        }
    return summary


def train_model(features: np.ndarray, labels: np.ndarray, seed: int):
    if RandomForestClassifier is not None and train_test_split is not None:
        x_train, x_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=0.25,
            random_state=seed,
            stratify=labels,
        )
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        val_acc = float(model.score(x_val, y_val))
        return model, val_acc, "RandomForestClassifier"

    # Fallback if sklearn is not present: sklearn-like nearest-centroid classifier.
    classes = sorted(set(labels.tolist()))
    centroids = {}
    for c in classes:
        centroids[c] = np.mean(features[labels == c], axis=0)

    model = NearestCentroidModel(centroids=centroids)
    preds = model.predict(features)
    acc = float(np.mean(preds == labels))

    return model, acc, "nearest_centroid_fallback"


def save_artifacts(
    output_model: Path,
    output_report: Path,
    output_ranges: Path,
    model_payload,
    feature_dim: int,
    report: dict,
    ranges_jsonable: dict,
):
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_ranges.parent.mkdir(parents=True, exist_ok=True)

    if joblib is None:
        raise RuntimeError("joblib is required to save model artifacts")

    joblib.dump(
        {
            "model": model_payload,
            "feature_columns": [f"f_{i}" for i in range(feature_dim)],
            "mdistortion_ranges": ranges_jsonable,
        },
        output_model,
    )

    with output_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with output_ranges.open("w", encoding="utf-8") as f:
        json.dump(ranges_jsonable, f, indent=2)


def main() -> None:
    args = parse_args()
    require_dependencies()

    yolo_model = YOLO(str(args.yolo_face_model))

    yolo_conf = float(args.yolo_conf)
    imgsz = int(args.imgsz)
    roi_pad = float(args.roi_pad)

    yolo_tuning_report = None
    if args.tune_yolo:
        faces_rows = read_faces_csv_rows(args.faces_csv, args.faces_images_dir)
        yolo_tuning_report = tune_yolo_hparams(
            model=yolo_model,
            faces_rows=faces_rows,
            seed=args.seed,
            iters=args.dfo_iterations,
        )
        if yolo_tuning_report:
            yolo_conf = yolo_tuning_report["best_conf"]
            imgsz = yolo_tuning_report["best_imgsz"]

    samples = gather_emotion_samples(args.emotions_dir, args.max_images_per_class)
    if not samples:
        raise RuntimeError(f"No images found under {args.emotions_dir}")

    detected_faces = gather_detected_faces(
        samples=samples,
        yolo_model=yolo_model,
        conf=yolo_conf,
        imgsz=imgsz,
        roi_pad=roi_pad,
    )
    if len(detected_faces) < 40:
        raise RuntimeError(
            "Too few detected faces to train reliably. Improve YOLO weights or increase dataset size."
        )

    node_opt = optimize_node_offsets(
        detected_faces=detected_faces,
        seed=args.seed,
        iters=args.de_iterations,
        popsize=args.de_popsize,
        f=args.de_f,
        cr=args.de_cr,
    )
    offsets = np.array(node_opt["offsets"], dtype=np.float32)

    features, labels, used_images = build_dataset(detected_faces, offsets)
    model, acc, model_name = train_model(features, labels, seed=args.seed)

    # "Ranges for every combination of Mdistortion matrix types": per-emotion
    # min/max/mean/std for each node-pair in the mouth/left-eye/right-eye matrices.
    ranges = compute_mdistortion_ranges(detected_faces, offsets)
    ranges_json = ranges_to_jsonable(ranges)
    range_acc = range_band_accuracy(detected_faces, offsets, ranges)

    report = {
        "n_samples_input": len(samples),
        "n_samples_detected": len(detected_faces),
        "n_samples_used": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "classes": sorted(set(labels.tolist())),
        "saved": {
            "model_weights": str(args.output_model),
            "report": str(args.output_report),
            "ranges": str(args.output_ranges),
        },
        "yolo": {
            "model": str(args.yolo_face_model),
            "conf": yolo_conf,
            "imgsz": imgsz,
            "roi_pad": roi_pad,
            "tuning": yolo_tuning_report,
        },
        "node_optimization": node_opt,
        "training": {
            "model": model_name,
            "accuracy": acc,
        },
        "mdistortion_ranges": {
            "groups": list(MDISTORTION_GROUPS.keys()),
            "summary": ranges_summary(ranges),
            "range_band_accuracy_in_sample": range_acc,
            "saved": str(args.output_ranges),
        },
        "first_images": used_images[:20],
    }

    save_artifacts(
        output_model=args.output_model,
        output_report=args.output_report,
        output_ranges=args.output_ranges,
        model_payload=model,
        feature_dim=int(features.shape[1]),
        report=report,
        ranges_jsonable=ranges_json,
    )

    print(f"Saved model weights: {args.output_model}")
    print(f"Saved training report: {args.output_report}")
    print(f"Saved Mdistortion ranges: {args.output_ranges}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
