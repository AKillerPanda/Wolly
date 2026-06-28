"""Live webcam demo built on the MediaPipe **Tasks** FaceLandmarker API.

Why this exists: the Python 3.14 build of mediapipe in this environment only
ships the Tasks API (`mediapipe.tasks`); the legacy `mediapipe.solutions`
FaceMesh/Pose used by detectors.py / main.py is not available. This module is
the Tasks-based replacement so you can see the node + Mdistortion pipeline run on
your camera today.

It detects 478 face landmarks, draws the same mouth / left-eye / right-eye node
groups the training pipeline uses, and shows live normalized Mdistortion metrics.

Run (window on your screen, press q to quit)::

    python -m affect_pi.live_tasks_demo --camera 0

Headless verification (no window; prints metrics, saves a snapshot)::

    python -m affect_pi.live_tasks_demo --headless --max-seconds 6
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .mdistortion_tasks import feature_from_pts

try:
    import joblib
except Exception:  # joblib only needed with --emotion-model
    joblib = None

# Same landmark groups the Mdistortion trainer uses (MediaPipe canonical IDs;
# the Tasks face mesh shares the 468+iris topology).
LEFT_EYE_SKIN_IDS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_SKIN_IDS = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
OUTER_MOUTH_IDS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

LEFT_CHEEK_ID = 234
RIGHT_CHEEK_ID = 454

GROUPS = {
    "mouth": (OUTER_MOUTH_IDS, (0, 255, 255)),       # yellow
    "left_eye": (LEFT_EYE_SKIN_IDS, (0, 255, 0)),    # green
    "right_eye": (RIGHT_EYE_SKIN_IDS, (0, 255, 0)),  # green
}

DEFAULT_MODEL = Path("models/face_landmarker.task")


def open_camera(index: int) -> tuple[cv2.VideoCapture | None, str | None]:
    for backend, name in [(cv2.CAP_DSHOW, "dshow"), (cv2.CAP_MSMF, "msmf"), (cv2.CAP_ANY, "default")]:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok = False
            for _ in range(8):
                ok, _ = cap.read()
                if ok:
                    break
            if ok:
                return cap, name
        cap.release()
    return None, None


def make_landmarker(model_path: Path) -> vision.FaceLandmarker:
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def points_for(ids: list[int], pts: np.ndarray) -> np.ndarray:
    return pts[np.array(ids, dtype=int)]


def normalized_mdistortion_energy(group_pts: np.ndarray, scale: float) -> float:
    """Mean pairwise distance of a node group, normalized by face scale.

    This is the live analogue of one Mdistortion matrix: the matrix the trainer
    builds is the full set of pairwise distances among these same nodes.
    """
    if len(group_pts) < 2 or scale <= 1e-6:
        return 0.0
    # Vectorized upper-triangle of the pairwise-distance matrix (no per-pair loop).
    diff = group_pts[:, None, :] - group_pts[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    iu = np.triu_indices(len(group_pts), k=1)
    return float(dist[iu].mean() / scale)


def open_ratio(top: np.ndarray, bottom: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    width = float(np.linalg.norm(left - right))
    if width <= 1e-6:
        return 0.0
    return float(np.linalg.norm(top - bottom) / width)


def compute_metrics(pts: np.ndarray) -> dict[str, float]:
    scale = float(np.linalg.norm(pts[LEFT_CHEEK_ID] - pts[RIGHT_CHEEK_ID]))
    m = {
        "mouth_md": normalized_mdistortion_energy(points_for(OUTER_MOUTH_IDS, pts), scale),
        "left_eye_md": normalized_mdistortion_energy(points_for(LEFT_EYE_SKIN_IDS, pts), scale),
        "right_eye_md": normalized_mdistortion_energy(points_for(RIGHT_EYE_SKIN_IDS, pts), scale),
        "mouth_open": open_ratio(pts[0], pts[17], pts[61], pts[291]),
        "left_eye_open": open_ratio(pts[159], pts[145], pts[33], pts[133]),
        "right_eye_open": open_ratio(pts[386], pts[374], pts[263], pts[362]),
    }
    return m


def draw(frame: np.ndarray, pts: np.ndarray, metrics: dict[str, float]) -> None:
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 1, (90, 90, 90), -1)
    for _name, (ids, color) in GROUPS.items():
        for i in ids:
            x, y = pts[i]
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    lines = [
        f"mouth Mdistortion: {metrics['mouth_md']:.3f}",
        f"left-eye Mdistortion: {metrics['left_eye_md']:.3f}",
        f"right-eye Mdistortion: {metrics['right_eye_md']:.3f}",
        f"mouth open: {metrics['mouth_open']:.3f}",
        f"eye open L/R: {metrics['left_eye_open']:.3f}/{metrics['right_eye_open']:.3f}",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


class LiveEmotionClassifier:
    """Loads an affect-pi emotion model and classifies live landmarks.

    Uses the same ``feature_from_pts`` + saved node offsets as training, and
    smooths probabilities over recent frames so the label does not flicker.
    """

    def __init__(self, model_path: Path | None, smooth: int = 7):
        self.enabled = False
        self.model = None
        self.offsets = np.zeros(6, dtype=np.float32)
        self.classes = None
        self.history: deque = deque(maxlen=smooth)
        if not model_path:
            return
        if joblib is None:
            raise SystemExit("joblib is required for --emotion-model (pip install joblib).")
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.offsets = np.array(payload.get("node_offsets", [0.0] * 6), dtype=np.float32)
        self.classes = payload.get("classes")
        self.enabled = True

    def predict(self, pts: np.ndarray) -> str | None:
        if not self.enabled:
            return None
        feat = feature_from_pts(pts, self.offsets)
        if feat is None:
            return None
        x = feat.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(x)[0], dtype=np.float32)
            self.history.append(probs)
            mean = np.mean(np.stack(self.history, axis=0), axis=0)
            idx = int(np.argmax(mean))
            classes = getattr(self.model, "classes_", self.classes)
            return f"{classes[idx]} {mean[idx]:.2f}"
        return str(self.model.predict(x)[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live MediaPipe Tasks FaceLandmarker + Mdistortion demo")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--mirror", action="store_true", help="Mirror the display horizontally")
    p.add_argument("--emotion-model", type=Path, default=None, help="joblib model from affect-pi-train-emotion for live classification")
    p.add_argument("--max-seconds", type=float, default=0.0, help="Auto-quit after N seconds (0 = run until q)")
    p.add_argument("--headless", action="store_true", help="No window; print metrics and save a snapshot")
    p.add_argument("--snapshot", type=Path, default=Path("artifacts/live_demo_snapshot.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(
            f"Model not found: {args.model}. Download face_landmarker.task into models/ first."
        )

    cap, backend = open_camera(args.camera)
    if cap is None:
        raise SystemExit(
            f"Could not open camera {args.camera}. Try a different index or check Windows camera privacy settings."
        )
    print(f"Camera opened on backend={backend}.")
    if not args.headless:
        print("Live window starting. Press 'q' in the window to quit.")

    landmarker = make_landmarker(args.model)
    classifier = LiveEmotionClassifier(args.emotion_model)
    if classifier.enabled:
        print(f"Emotion model loaded: {args.emotion_model} (classes: {classifier.classes})")
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    frames = 0
    face_frames = 0
    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            frames += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            h, w = frame.shape[:2]
            if result.face_landmarks:
                face_frames += 1
                lm = result.face_landmarks[0]
                pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
                metrics = compute_metrics(pts)
                draw(frame, pts, metrics)
                emotion = classifier.predict(pts)
                if emotion:
                    cv2.putText(frame, f"emotion: {emotion}", (10, h - 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"FACE ({len(pts)} nodes)", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                now = time.time()
                if now - last_print > 0.5:
                    fps = frames / (now - start)
                    tail = f" emotion={emotion}" if emotion else ""
                    print(f"[{frames:4d}] fps={fps:4.1f} mouth_md={metrics['mouth_md']:.3f} "
                          f"eye_open L/R={metrics['left_eye_open']:.3f}/{metrics['right_eye_open']:.3f}{tail}")
                    last_print = now
            else:
                cv2.putText(frame, "no face", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            if not args.headless:
                cv2.imshow("affect-pi live (MediaPipe Tasks)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if args.max_seconds > 0 and (time.time() - start) >= args.max_seconds:
                break
    finally:
        cv2.imwrite(str(args.snapshot), frame)
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        landmarker.close()

    elapsed = max(time.time() - start, 1e-6)
    print(f"\nDone. frames={frames} face_frames={face_frames} "
          f"avg_fps={frames / elapsed:.1f} snapshot={args.snapshot}")


if __name__ == "__main__":
    main()
