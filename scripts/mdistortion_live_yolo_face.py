"""
mdistortion_live_yolo_face.py

Live webcam pipeline:
1) YOLO detects/tracks the face.
2) The YOLO face box becomes the search space / ROI.
3) MediaPipe Face Mesh finds facial feature geometry inside that ROI.
4) The script places nodes around:
   - eye skin / eyelid area, avoiding the eyeball/iris
   - outer mouth/lip area, avoiding the inner mouth
   - brow + forehead skin area
5) It calculates normalized Mdistortions for mouth, eyes, and forehead.
6) Optional: save live Mdistortion rows to CSV for training your emotion classifier.

Install:
    pip install ultralytics opencv-python mediapipe numpy pandas joblib

Run:
    python mdistortion_live_yolo_face.py --yolo-face-model runs/detect/face_detector/weights/best.pt --camera 0

Optional CSV logging:
    python mdistortion_live_yolo_face.py --yolo-face-model runs/detect/face_detector/weights/best.pt --camera 0 --save-csv mdistortions_live.csv

Controls while running:
    q  = quit
    b  = set neutral baseline from the most recent frames
    r  = reset neutral baseline
    l  = toggle CSV logging on/off, if --save-csv was provided

Important:
    Your YOLO model should be trained to detect a face as a bounding box.
    This script uses YOLO for the face search space and MediaPipe Face Mesh for the nodes.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

try:
    import joblib
except Exception:  # joblib is only needed if --emotion-model is used
    joblib = None


Point = Tuple[float, float]
FeatureDict = Dict[str, float]
NodeDict = Dict[str, Point]


# -----------------------------------------------------------------------------
# MediaPipe Face Mesh landmark choices
# -----------------------------------------------------------------------------
# These are MediaPipe Face Mesh landmark IDs. The selected points are focused on
# skin / boundary regions, not eyeballs or inner mouth.
#
# Eye nodes: eyelid / eye contour / periocular skin. No iris/eyeball landmarks.
# Mouth nodes: outer lip contour only. No inner mouth landmarks.
# Forehead nodes: brow landmarks + upper-face landmarks + synthetic forehead
#                 points projected above the brows.
# -----------------------------------------------------------------------------

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

LEFT_BROW_IDS = [70, 63, 105, 66, 107]
RIGHT_BROW_IDS = [336, 296, 334, 293, 300]
UPPER_FACE_IDS = [10, 151, 9, 8, 168]

# Stable scale landmarks for normalization.
LEFT_CHEEK_ID = 234
RIGHT_CHEEK_ID = 454
CHIN_ID = 152
TOP_FACE_ID = 10
NOSE_BRIDGE_ID = 168


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 0.0
    track_id: Optional[int] = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class YoloFaceTracker:
    """YOLO-based face detector/tracker.

    If YOLO tracking works in your environment, it uses model.track(...). If not,
    it falls back to model.predict(...). The script then uses the largest box as
    the active face unless a tracked ID is available.
    """

    def __init__(self, model_path: str, conf: float, imgsz: int, use_tracking: bool = True):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.use_tracking = use_tracking

    def get_face_boxes(self, frame: np.ndarray) -> List[FaceBox]:
        try:
            if self.use_tracking:
                results = self.model.track(
                    frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            else:
                results = self.model.predict(
                    frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    verbose=False,
                )
        except Exception:
            # Fallback if tracking config is unavailable.
            results = self.model.predict(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
            )

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(xyxy))

        ids = None
        if getattr(result.boxes, "id", None) is not None:
            try:
                ids = result.boxes.id.cpu().numpy().astype(int)
            except Exception:
                ids = None

        boxes: List[FaceBox] = []
        h, w = frame.shape[:2]

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(int).tolist()
            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            track_id = int(ids[i]) if ids is not None and i < len(ids) else None
            boxes.append(FaceBox(x1, y1, x2, y2, float(confs[i]), track_id))

        return boxes

    @staticmethod
    def select_primary_face(boxes: List[FaceBox]) -> Optional[FaceBox]:
        if not boxes:
            return None
        return max(boxes, key=lambda b: b.area * max(b.conf, 1e-3))


class FaceNodeExtractor:
    """Extracts facial skin nodes inside a YOLO face ROI using MediaPipe Face Mesh."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )

    @staticmethod
    def _pad_box(box: FaceBox, frame_shape: Tuple[int, int, int], pad_ratio: float) -> FaceBox:
        h, w = frame_shape[:2]
        pad_x = int(box.width * pad_ratio)
        pad_y = int(box.height * pad_ratio)

        return FaceBox(
            x1=max(0, box.x1 - pad_x),
            y1=max(0, box.y1 - pad_y),
            x2=min(w - 1, box.x2 + pad_x),
            y2=min(h - 1, box.y2 + pad_y),
            conf=box.conf,
            track_id=box.track_id,
        )

    @staticmethod
    def _landmarks_to_points(
        landmarks,
        roi_box: FaceBox,
        roi_w: int,
        roi_h: int,
    ) -> Dict[int, Point]:
        pts: Dict[int, Point] = {}
        for idx, lm in enumerate(landmarks):
            x = roi_box.x1 + lm.x * roi_w
            y = roi_box.y1 + lm.y * roi_h
            pts[idx] = (float(x), float(y))
        return pts

    @staticmethod
    def _add_group_nodes(
        nodes: NodeDict,
        all_pts: Dict[int, Point],
        group_prefix: str,
        ids: List[int],
    ) -> None:
        for local_i, lm_id in enumerate(ids):
            if lm_id in all_pts:
                nodes[f"{group_prefix}_{local_i:02d}_lm{lm_id}"] = all_pts[lm_id]

    @staticmethod
    def _add_synthetic_forehead_nodes(nodes: NodeDict, all_pts: Dict[int, Point]) -> None:
        """Create extra forehead-skin points above the brows.

        MediaPipe Face Mesh gives very useful brow/upper-face points, but not a
        dense full forehead wrinkle map. These synthetic points are projected
        upward from brow points so your foreheadMdistortions include the skin
        region that scrunches during expressions.
        """
        required = [CHIN_ID, TOP_FACE_ID]
        if any(i not in all_pts for i in required):
            return

        face_height = euclidean(all_pts[TOP_FACE_ID], all_pts[CHIN_ID])
        if face_height < 1e-6:
            return

        # Up direction in image coordinates is negative y.
        lift_1 = 0.055 * face_height
        lift_2 = 0.105 * face_height

        brow_ids = LEFT_BROW_IDS + RIGHT_BROW_IDS
        for i, lm_id in enumerate(brow_ids):
            if lm_id not in all_pts:
                continue
            x, y = all_pts[lm_id]
            nodes[f"forehead_synth_low_{i:02d}_from_lm{lm_id}"] = (x, y - lift_1)
            nodes[f"forehead_synth_high_{i:02d}_from_lm{lm_id}"] = (x, y - lift_2)

    def extract_nodes(
        self,
        frame_bgr: np.ndarray,
        face_box: FaceBox,
        pad_ratio: float = 0.15,
    ) -> Tuple[Optional[NodeDict], Optional[Dict[int, Point]], Optional[FaceBox]]:
        """Return named nodes, raw landmark points, and padded ROI box."""
        roi_box = self._pad_box(face_box, frame_bgr.shape, pad_ratio)
        roi = frame_bgr[roi_box.y1:roi_box.y2, roi_box.x1:roi_box.x2]

        if roi.size == 0:
            return None, None, None

        roi_h, roi_w = roi.shape[:2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(roi_rgb)

        if not result.multi_face_landmarks:
            return None, None, roi_box

        landmarks = result.multi_face_landmarks[0].landmark
        all_pts = self._landmarks_to_points(landmarks, roi_box, roi_w, roi_h)

        nodes: NodeDict = {}

        self._add_group_nodes(nodes, all_pts, "left_eye_skin", LEFT_EYE_SKIN_IDS)
        self._add_group_nodes(nodes, all_pts, "right_eye_skin", RIGHT_EYE_SKIN_IDS)
        self._add_group_nodes(nodes, all_pts, "outer_mouth", OUTER_MOUTH_IDS)
        self._add_group_nodes(nodes, all_pts, "left_brow", LEFT_BROW_IDS)
        self._add_group_nodes(nodes, all_pts, "right_brow", RIGHT_BROW_IDS)
        self._add_group_nodes(nodes, all_pts, "upper_face", UPPER_FACE_IDS)
        self._add_synthetic_forehead_nodes(nodes, all_pts)

        return nodes, all_pts, roi_box


class MdistortionEngine:
    """Calculates normalized facial Mdistortion features from named nodes."""

    GROUP_PREFIXES = {
        "mouthMdistortions": ["outer_mouth"],
        "leftEyeMdistortions": ["left_eye_skin"],
        "rightEyeMdistortions": ["right_eye_skin"],
        "foreheadMdistortions": ["left_brow", "right_brow", "upper_face", "forehead_synth"],
    }

    def __init__(self):
        self.baseline: Optional[FeatureDict] = None
        self.recent_features: deque[FeatureDict] = deque(maxlen=45)

    @staticmethod
    def face_scale(all_pts: Optional[Dict[int, Point]], face_box: FaceBox) -> float:
        """A stable denominator so movement is not confused with camera distance."""
        if all_pts is not None and LEFT_CHEEK_ID in all_pts and RIGHT_CHEEK_ID in all_pts:
            cheek_width = euclidean(all_pts[LEFT_CHEEK_ID], all_pts[RIGHT_CHEEK_ID])
            if cheek_width > 1e-6:
                return cheek_width

        # Fallback to YOLO face box width.
        return max(float(face_box.width), 1.0)

    @staticmethod
    def _nodes_for_prefixes(nodes: NodeDict, prefixes: List[str]) -> NodeDict:
        selected: NodeDict = {}
        for name, pt in nodes.items():
            if any(name.startswith(prefix) for prefix in prefixes):
                selected[name] = pt
        return selected

    @staticmethod
    def _pairwise_group_features(group_name: str, group_nodes: NodeDict, scale: float) -> FeatureDict:
        names = sorted(group_nodes.keys())
        if len(names) < 2:
            return {}
        # One vectorized pairwise-distance matrix instead of a per-pair loop.
        pts = np.array([group_nodes[n] for n in names], dtype=np.float32)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)) / scale
        iu = np.triu_indices(len(names), k=1)
        return {f"{group_name}:{names[i]}__to__{names[j]}": float(dist[i, j])
                for i, j in zip(iu[0].tolist(), iu[1].tolist())}

    @staticmethod
    def _special_expression_features(nodes: NodeDict, all_pts: Dict[int, Point], scale: float) -> FeatureDict:
        """Small set of interpretable features that are useful for live display/training."""
        f: FeatureDict = {}

        def lm_dist(name: str, a: int, b: int) -> None:
            if a in all_pts and b in all_pts:
                f[name] = euclidean(all_pts[a], all_pts[b]) / scale

        # Mouth shape: outer mouth only, avoiding inner mouth.
        lm_dist("special:mouth_width_61_291", 61, 291)
        lm_dist("special:mouth_open_outer_0_17", 0, 17)
        if "special:mouth_width_61_291" in f and f["special:mouth_width_61_291"] > 1e-6:
            f["special:mouth_open_ratio"] = f.get("special:mouth_open_outer_0_17", 0.0) / f["special:mouth_width_61_291"]

        # Eye openness from eyelid top/bottom points. These avoid iris/eyeball.
        lm_dist("special:left_eye_open_159_145", 159, 145)
        lm_dist("special:left_eye_width_33_133", 33, 133)
        if "special:left_eye_width_33_133" in f and f["special:left_eye_width_33_133"] > 1e-6:
            f["special:left_eye_open_ratio"] = f.get("special:left_eye_open_159_145", 0.0) / f["special:left_eye_width_33_133"]

        lm_dist("special:right_eye_open_386_374", 386, 374)
        lm_dist("special:right_eye_width_263_362", 263, 362)
        if "special:right_eye_width_263_362" in f and f["special:right_eye_width_263_362"] > 1e-6:
            f["special:right_eye_open_ratio"] = f.get("special:right_eye_open_386_374", 0.0) / f["special:right_eye_width_263_362"]

        # Brow / forehead scrunch proxies.
        lm_dist("special:left_brow_to_eye_105_159", 105, 159)
        lm_dist("special:right_brow_to_eye_334_386", 334, 386)
        lm_dist("special:brow_inner_distance_107_336", 107, 336)
        lm_dist("special:forehead_height_10_168", 10, 168)

        return f

    def calculate(
        self,
        nodes: NodeDict,
        all_pts: Dict[int, Point],
        face_box: FaceBox,
    ) -> FeatureDict:
        scale = self.face_scale(all_pts, face_box)
        features: FeatureDict = {}

        for group_name, prefixes in self.GROUP_PREFIXES.items():
            group_nodes = self._nodes_for_prefixes(nodes, prefixes)
            features.update(self._pairwise_group_features(group_name, group_nodes, scale))

        features.update(self._special_expression_features(nodes, all_pts, scale))

        self.recent_features.append(features)

        if self.baseline is not None:
            for k, v in list(features.items()):
                if k in self.baseline:
                    features[f"delta:{k}"] = v - self.baseline[k]

        return features

    def set_baseline_from_recent(self) -> bool:
        if not self.recent_features:
            return False

        keys = sorted(set().union(*(f.keys() for f in self.recent_features)))
        baseline: FeatureDict = {}

        for k in keys:
            vals = [f[k] for f in self.recent_features if k in f]
            if vals:
                baseline[k] = float(np.mean(vals))

        self.baseline = baseline
        return True

    def reset_baseline(self) -> None:
        self.baseline = None


class OptionalEmotionClassifier:
    """Optional hook for a joblib classifier trained on these feature names.

    Expected joblib format:
        {
            "model": sklearn_model,
            "feature_columns": [feature_name_1, feature_name_2, ...]
        }
    """

    def __init__(self, model_path: Optional[str]):
        self.enabled = False
        self.model = None
        self.feature_columns: List[str] = []
        self.prob_history: deque[np.ndarray] = deque(maxlen=7)

        if not model_path:
            return

        if joblib is None:
            raise RuntimeError("joblib is not installed. Install with: pip install joblib")

        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_columns = list(payload["feature_columns"])
        self.enabled = True

    def predict_label(self, features: FeatureDict) -> Optional[str]:
        if not self.enabled or self.model is None:
            return None

        x = np.array([[features.get(col, 0.0) for col in self.feature_columns]], dtype=np.float32)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            self.prob_history.append(probs)
            probs = np.mean(np.array(self.prob_history), axis=0)
            idx = int(np.argmax(probs))
            return f"{self.model.classes_[idx]} {probs[idx]:.2f}"

        pred = self.model.predict(x)[0]
        return str(pred)


class CsvLogger:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.active = bool(path)
        self.file = None
        self.writer = None
        self.header_written = False
        self.fieldnames: List[str] = []

    def toggle(self) -> bool:
        if self.path is None:
            return False
        self.active = not self.active
        return self.active

    def write(self, label: str, features: FeatureDict, face_box: FaceBox) -> None:
        if self.path is None or not self.active:
            return

        if self.file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("w", newline="", encoding="utf-8")

        base = {
            "timestamp": time.time(),
            "label": label,
            "face_x1": face_box.x1,
            "face_y1": face_box.y1,
            "face_x2": face_box.x2,
            "face_y2": face_box.y2,
            "face_conf": face_box.conf,
            "track_id": face_box.track_id if face_box.track_id is not None else "",
        }

        row = {**base, **features}

        if not self.header_written:
            self.fieldnames = list(row.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self.header_written = True

        # Keep stable columns. New unseen columns are ignored after header creation.
        stable_row = {k: row.get(k, "") for k in self.fieldnames}
        assert self.writer is not None
        self.writer.writerow(stable_row)
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def euclidean(a: Point, b: Point) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def draw_node_group(frame: np.ndarray, nodes: NodeDict, prefix: str, color: Tuple[int, int, int]) -> None:
    for name, (x, y) in nodes.items():
        if not name.startswith(prefix):
            continue
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)


def draw_all_nodes(frame: np.ndarray, nodes: NodeDict) -> None:
    # OpenCV color order is BGR.
    draw_node_group(frame, nodes, "outer_mouth", (0, 255, 255))       # yellow
    draw_node_group(frame, nodes, "left_eye_skin", (0, 255, 0))      # green
    draw_node_group(frame, nodes, "right_eye_skin", (0, 255, 0))     # green
    draw_node_group(frame, nodes, "left_brow", (255, 0, 255))        # magenta
    draw_node_group(frame, nodes, "right_brow", (255, 0, 255))       # magenta
    draw_node_group(frame, nodes, "upper_face", (255, 255, 0))       # cyan
    draw_node_group(frame, nodes, "forehead_synth", (255, 255, 255)) # white


def put_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def get_display_values(features: FeatureDict) -> List[str]:
    wanted = [
        "special:mouth_open_ratio",
        "special:left_eye_open_ratio",
        "special:right_eye_open_ratio",
        "special:brow_inner_distance_107_336",
        "special:left_brow_to_eye_105_159",
        "special:right_brow_to_eye_334_386",
    ]

    lines = []
    for k in wanted:
        if k in features:
            short = k.replace("special:", "")
            lines.append(f"{short}: {features[k]:.3f}")

    # Show deltas too, if baseline is active.
    delta_wanted = [f"delta:{k}" for k in wanted]
    for k in delta_wanted:
        if k in features:
            short = k.replace("delta:special:", "d_")
            lines.append(f"{short}: {features[k]:+.3f}")

    return lines[:10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO face ROI + facial skin nodes + live Mdistortions")
    parser.add_argument(
        "--yolo-face-model",
        type=str,
        default="runs/detect/face_detector/weights/best.pt",
        help="Path to your YOLO face detection model weights, e.g. runs/detect/face_detector/weights/best.pt",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Try 1 or 2 if 0 does not work.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--no-track", action="store_true", help="Use YOLO predict instead of YOLO track")
    parser.add_argument("--roi-pad", type=float, default=0.15, help="Padding around YOLO face ROI before landmarks")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional CSV path for saving live Mdistortion features")
    parser.add_argument("--csv-label", type=str, default="unlabeled", help="Label written into CSV rows, e.g. happy, angry, neutral")
    parser.add_argument("--emotion-model", type=str, default=None, help="Optional joblib emotion classifier trained on these feature names")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam display horizontally")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    face_tracker = YoloFaceTracker(
        model_path=args.yolo_face_model,
        conf=args.conf,
        imgsz=args.imgsz,
        use_tracking=not args.no_track,
    )
    node_extractor = FaceNodeExtractor()
    md_engine = MdistortionEngine()
    classifier = OptionalEmotionClassifier(args.emotion_model)
    csv_logger = CsvLogger(args.save_csv)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Try --camera 1 or check webcam permissions."
        )

    print("Running live YOLO face ROI + Mdistortions.")
    print("Controls: q=quit, b=set baseline, r=reset baseline, l=toggle csv logging")

    fps_times: deque[float] = deque(maxlen=30)

    try:
        while True:
            frame_start = time.time()
            ok, frame = cap.read()
            if not ok:
                print("Could not read webcam frame.")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            boxes = face_tracker.get_face_boxes(frame)
            face_box = face_tracker.select_primary_face(boxes)

            if face_box is None:
                put_text(frame, "No YOLO face box found", 20, 35, color=(0, 0, 255), thickness=2)
            else:
                cv2.rectangle(frame, (face_box.x1, face_box.y1), (face_box.x2, face_box.y2), (255, 0, 0), 2)

                track_text = f"face conf={face_box.conf:.2f}"
                if face_box.track_id is not None:
                    track_text += f" id={face_box.track_id}"
                put_text(frame, track_text, face_box.x1, max(25, face_box.y1 - 8), color=(255, 0, 0), thickness=2)

                nodes, all_pts, roi_box = node_extractor.extract_nodes(frame, face_box, pad_ratio=args.roi_pad)

                if roi_box is not None:
                    cv2.rectangle(frame, (roi_box.x1, roi_box.y1), (roi_box.x2, roi_box.y2), (80, 80, 255), 1)

                if nodes is None or all_pts is None:
                    put_text(frame, "Face ROI found, but no facial nodes found", 20, 65, color=(0, 0, 255), thickness=2)
                else:
                    draw_all_nodes(frame, nodes)
                    features = md_engine.calculate(nodes, all_pts, face_box)
                    csv_logger.write(args.csv_label, features, face_box)

                    prediction = classifier.predict_label(features)
                    if prediction:
                        put_text(frame, f"expression: {prediction}", 20, 35, color=(0, 255, 255), thickness=2)

                    display_lines = get_display_values(features)
                    y = 65
                    for line in display_lines:
                        put_text(frame, line, 20, y, color=(255, 255, 255), thickness=1)
                        y += 22

                    baseline_status = "baseline: ON" if md_engine.baseline is not None else "baseline: OFF"
                    put_text(frame, baseline_status, 20, y + 5, color=(0, 255, 255), thickness=1)

            # FPS display.
            fps_times.append(time.time() - frame_start)
            avg_dt = float(np.mean(fps_times)) if fps_times else 0.0
            fps = 1.0 / avg_dt if avg_dt > 1e-6 else 0.0
            put_text(frame, f"FPS: {fps:.1f}", 20, frame.shape[0] - 20, color=(255, 255, 255), thickness=1)

            if args.save_csv:
                csv_state = "CSV ON" if csv_logger.active else "CSV OFF"
                put_text(frame, csv_state, frame.shape[1] - 130, frame.shape[0] - 20, color=(255, 255, 255), thickness=1)

            cv2.imshow("YOLO Face ROI -> Skin Nodes -> Mdistortions", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("b"):
                ok_baseline = md_engine.set_baseline_from_recent()
                print("Neutral baseline set." if ok_baseline else "No recent features available for baseline.")
            if key == ord("r"):
                md_engine.reset_baseline()
                print("Neutral baseline reset.")
            if key == ord("l"):
                active = csv_logger.toggle()
                if args.save_csv:
                    print(f"CSV logging {'ON' if active else 'OFF'}: {args.save_csv}")
                else:
                    print("CSV path not provided. Use --save-csv mdistortions_live.csv")

    finally:
        csv_logger.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
