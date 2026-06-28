#!/usr/bin/env python3
"""
live_emotion_eyes.py - Glue app: camera -> emotion -> dual robot-eye displays.

This is integration step #1: it joins the two halves of the repo into one loop
that you can run in VS Code with your webcam and watch the cute eyes react.

    Raspberry Pi / webcam frame
        -> MediaPipe Tasks FaceLandmarker  (src/affect_pi)
        -> Mdistortion feature vector + trained emotion model
        -> map emotion -> eye Mood
        -> EyeController + EyeRenderer       (src/robot_eyes)
        -> three OpenCV windows: Left Eye | Right Eye | Camera preview

No pygame and no hardware are needed: the eye renderer is pure numpy, and we
present its framebuffers with cv2.imshow (which trivially supports separate
windows). This script is the desktop simulator path -- it renders straight to
OpenCV windows. Deploying to the real ST7789 panels would route the same
EyeController/EyeRenderer output through robot_eyes.backends.SpiBackend instead
(that wiring is not done in this script yet).

Run (defaults assume you ran affect-pi-train-emotion already)::

    python scripts/live_emotion_eyes.py --mirror

Verify the eyes without a camera (renders every mood, incl. new Sad/Fear, to a
PNG contact sheet)::

    python scripts/live_emotion_eyes.py --selftest

Keys in the windows: q / Esc quit, space = force a blink.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

# --- make the in-repo packages importable even without the editable install. ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # -> import affect_pi, robot_eyes

from affect_pi.mdistortion_tasks import (  # noqa: E402
    DEFAULT_MODEL,
    detect_video,
    feature_from_pts,
    load_landmarker,
)
from affect_pi.train_mdistortion_de import (  # noqa: E402
    LEFT_EYE_SKIN_IDS,
    OUTER_MOUTH_IDS,
    RIGHT_EYE_SKIN_IDS,
)
from affect_pi.face_identity import FaceRegistry, IdentityTracker  # noqa: E402
from robot_eyes.behavior import EMOTES, AdaptivePolicy, Emote  # noqa: E402
from robot_eyes.config import Config, Mood  # noqa: E402
from robot_eyes.controller import EyeController  # noqa: E402
from robot_eyes.renderer import EyeRenderer, RenderState  # noqa: E402

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

DEFAULT_EMOTION_MODEL = Path("artifacts/emotion_tasks_model.joblib")
DEFAULT_FACES_FILE = Path("artifacts/known_faces.txt")

# Trained classes are Angry/Fear/Happy/Sad/Suprise (note dataset's spelling).
# Each maps to a distinct eye Mood. Anything unmapped / no face -> neutral idle.
EMOTION_TO_MOOD = {
    "Happy": Mood.HAPPY,
    "Angry": Mood.ANGRY,
    "Sad": Mood.SAD,
    "Fear": Mood.FEAR,
    "Suprise": Mood.SURPRISED,
    "Surprise": Mood.SURPRISED,  # tolerate the correctly-spelled variant too
}

# Node groups to draw on the camera preview (same IDs the trainer uses).
PREVIEW_GROUPS = {
    "mouth": (OUTER_MOUTH_IDS, (0, 255, 255)),      # yellow (BGR)
    "left_eye": (LEFT_EYE_SKIN_IDS, (0, 255, 0)),   # green
    "right_eye": (RIGHT_EYE_SKIN_IDS, (0, 255, 0)),
}


# --------------------------------------------------------------------------- #
#  Emotion classifier wrapper (smoothed)                                      #
# --------------------------------------------------------------------------- #
class EmotionReader:
    """Loads the trained joblib model and turns landmarks into a smoothed label."""

    def __init__(self, model_path: Path, smooth: int = 7):
        if joblib is None:
            raise SystemExit("joblib is required (pip install joblib).")
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.offsets = np.asarray(payload.get("node_offsets", [0.0] * 6), dtype=np.float32)
        self.classes = list(getattr(self.model, "classes_", payload.get("classes", [])))
        self._idx = {str(c): i for i, c in enumerate(self.classes)}
        self.history: deque[np.ndarray] = deque(maxlen=smooth)
        self.last_mean: np.ndarray | None = None

    def predict(self, pts: np.ndarray) -> tuple[str | None, float]:
        feat = feature_from_pts(pts, self.offsets)
        if feat is None:
            return None, 0.0
        x = feat.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(x)[0], dtype=np.float32)
            self.history.append(probs)
            mean = np.mean(np.stack(self.history, axis=0), axis=0)
            self.last_mean = mean
            idx = int(np.argmax(mean))
            return str(self.classes[idx]), float(mean[idx])
        return str(self.model.predict(x)[0]), 1.0

    def happiness(self) -> float:
        """Scalar 'how happy does the user look' in roughly [-1, 1] from the last
        smoothed probabilities. Used as the reward for the adaptive eye policy."""
        if self.last_mean is None:
            return 0.0
        def p(name: str) -> float:
            i = self._idx.get(name)
            return float(self.last_mean[i]) if i is not None else 0.0
        return p("Happy") + 0.3 * p("Suprise") - 0.5 * (p("Sad") + p("Angry") + p("Fear"))


# --------------------------------------------------------------------------- #
#  Camera helpers                                                             #
# --------------------------------------------------------------------------- #
def open_camera(index: int):
    """Open a webcam, trying Windows-friendly backends first, then fall back."""
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


def eye_to_bgr(fb_rgb: np.ndarray, scale: int) -> np.ndarray:
    """Renderer outputs (H, W, 3) RGB; convert to BGR and pixel-upscale for display."""
    bgr = cv2.cvtColor(fb_rgb, cv2.COLOR_RGB2BGR)
    if scale > 1:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return bgr


def start_emote(controller: EyeController, emote: Emote, user_label: str | None) -> None:
    """Apply an emote's mood + one-shot motion to the controller."""
    controller.set_mood(emote.mood or EMOTION_TO_MOOD.get(user_label, Mood.NEUTRAL))
    if emote.action == "blink":
        controller.blink()
    elif emote.action == "wink_left":
        controller.wink("left")
    elif emote.action == "wink_right":
        controller.wink("right")
    elif emote.action == "look":
        controller.look(emote.look[0], emote.look[1], hold=False)


def draw_preview(frame: np.ndarray, pts: np.ndarray | None, label: str | None,
                 conf: float, mood: Mood, fps: float, who: str | None = None,
                 emote_text: str | None = None) -> None:
    h = frame.shape[0]
    if pts is not None:
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 1, (90, 90, 90), -1)
        for _name, (ids, color) in PREVIEW_GROUPS.items():
            for i in ids:
                if i < len(pts):
                    x, y = pts[i]
                    cv2.circle(frame, (int(x), int(y)), 2, color, -1)
        text = f"{label} {conf:.2f}" if label else "face (no label)"
        cv2.putText(frame, f"emotion: {text}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        if who is not None:
            cv2.putText(frame, f"who: {who}", (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "no face", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    if emote_text is not None:
        cv2.putText(frame, f"emote: {emote_text}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"eye mood: {mood.value}", (10, h - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"fps {fps:4.1f}  (q quit, space blink)", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
#  Self-test (no camera): render every mood to a PNG contact sheet            #
# --------------------------------------------------------------------------- #
def selftest(cfg: Config, out: Path, scale: int) -> None:
    renderer = EyeRenderer(cfg)
    from robot_eyes.renderer import RenderState
    tiles = []
    for mood in Mood:
        left = eye_to_bgr(renderer.render(RenderState(mood=mood), "left"), scale)
        right = eye_to_bgr(renderer.render(RenderState(mood=mood), "right"), scale)
        pair = np.hstack([left, np.full((left.shape[0], 8, 3), 40, np.uint8), right])
        cv2.putText(pair, mood.value, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2, cv2.LINE_AA)
        tiles.append(pair)
    sheet = np.vstack([np.pad(t, ((0, 12), (0, 0), (0, 0)), constant_values=0) for t in tiles])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"Rendered {len(tiles)} moods (incl. sad + fear) -> {out}")


# --------------------------------------------------------------------------- #
#  Main loop                                                                  #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live camera -> emotion -> dual robot eyes (OpenCV sim).")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--landmarker-model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--emotion-model", type=Path, default=DEFAULT_EMOTION_MODEL)
    p.add_argument("--mirror", action="store_true", help="Mirror the camera horizontally")
    p.add_argument("--eye-scale", type=int, default=2, help="Integer upscale for the eye windows")
    p.add_argument("--min-conf", type=float, default=0.40,
                   help="Below this smoothed confidence the eyes fall back to neutral idle")
    p.add_argument("--faces-file", type=Path, default=DEFAULT_FACES_FILE,
                   help="Text file of known faces (recognise + auto-enroll)")
    p.add_argument("--match-threshold", type=float, default=0.045,
                   help="Max signature distance to count as the same face")
    p.add_argument("--no-enroll", action="store_true",
                   help="Recognise only; do not remember new faces")
    p.add_argument("--epsilon", type=float, default=0.2,
                   help="Exploration rate: fraction of emotes that are a random trial")
    p.add_argument("--policy-file", type=Path, default=Path("artifacts/emote_policy.txt"),
                   help="Where the learned per-person emote preferences persist")
    p.add_argument("--no-learn", action="store_true",
                   help="Use learned preferences but do not keep updating them")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for blinks/saccades")
    p.add_argument("--selftest", action="store_true", help="No camera; render every mood to a PNG and exit")
    p.add_argument("--selftest-out", type=Path, default=Path("artifacts/eyes_moods_preview.png"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config()

    if args.selftest:
        selftest(cfg, args.selftest_out, max(args.eye_scale, 2))
        return 0

    if not args.landmarker_model.exists():
        raise SystemExit(f"FaceLandmarker model not found: {args.landmarker_model}. "
                         "Download face_landmarker.task into models/ first (see README).")
    if not args.emotion_model.exists():
        raise SystemExit(f"Emotion model not found: {args.emotion_model}. "
                         "Train one first: affect-pi-train-emotion")

    landmarker = load_landmarker(args.landmarker_model, video=True)
    reader = EmotionReader(args.emotion_model)
    renderer = EyeRenderer(cfg)
    controller = EyeController(cfg, seed=args.seed)
    registry = FaceRegistry.load(args.faces_file)
    identity = IdentityTracker(registry, match_threshold=args.match_threshold,
                               auto_enroll=not args.no_enroll)
    policy = AdaptivePolicy(EMOTES, epsilon=args.epsilon, seed=args.seed,
                            path=None if args.no_learn else args.policy_file)
    print(f"Emotion classes: {reader.classes}")
    print(f"Known faces ({args.faces_file}): {[r.label for r in registry.records] or 'none yet'}")
    print(f"Emotes: {[e.name for e in EMOTES]}  (epsilon={args.epsilon})")

    cap, backend = open_camera(args.camera)
    if cap is None:
        raise SystemExit(f"Could not open camera {args.camera}. Try a different --camera index "
                         "or check Windows camera privacy settings.")
    print(f"Camera opened (backend={backend}). Three windows will appear. Press q to quit.")

    # Lay the windows out side by side so they do not stack on top of each other.
    cv2.namedWindow("Left Eye", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Right Eye", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
    ew = cfg.screen.width * args.eye_scale
    cv2.moveWindow("Left Eye", 40, 80)
    cv2.moveWindow("Right Eye", 40 + ew + 20, 80)
    cv2.moveWindow("Camera", 40 + 2 * (ew + 20), 80)

    start = time.time()
    last = time.perf_counter()
    last_ts = -1
    fps_hist: deque[float] = deque(maxlen=30)

    # Adaptive emote cycle: pick an emote, hold it for its duration while gathering
    # the user's happiness, then score it and pick the next one.
    cur_emote: Emote | None = None
    emote_elapsed = 0.0
    reward_samples: list[float] = []
    arm_identity = "anon"
    arm_context = "neutral"
    explored = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - last
            last = now
            fps_hist.append(dt)
            fps = 1.0 / (sum(fps_hist) / len(fps_hist)) if fps_hist else 0.0

            ts = int((time.time() - start) * 1000)
            if ts <= last_ts:           # Tasks VIDEO mode needs strictly increasing ts
                ts = last_ts + 1
            last_ts = ts
            pts = detect_video(landmarker, frame, ts)

            # Recognise / remember the face internally (drives per-person behaviour
            # later). We do NOT show a name -- just whether it knows this is the
            # same person it has seen before.
            identity.update(pts)
            if identity.last_label is not None:
                who = "known"
            elif pts is not None and not args.no_enroll:
                who = "learning face"
            elif pts is not None:
                who = "unknown"
            else:
                who = None

            label, conf = (None, 0.0)
            if pts is not None:
                label, conf = reader.predict(pts)

            # --- adaptive emote cycle ---
            # Context = the user's current mood (only when confident); reward =
            # how happy they look while an emote is on screen.
            if pts is not None:
                reward_samples.append(reader.happiness())
            emote_elapsed += dt

            if cur_emote is None or emote_elapsed >= cur_emote.duration:
                # Score the emote that just finished, then choose the next.
                if cur_emote is not None and reward_samples and not args.no_learn:
                    policy.update(arm_identity, arm_context, cur_emote, float(np.mean(reward_samples)))
                arm_identity = identity.last_label or "anon"
                arm_context = label if (label is not None and conf >= args.min_conf) else "neutral"
                cur_emote, explored = policy.select(arm_identity, arm_context)
                start_emote(controller, cur_emote, label)
                emote_elapsed = 0.0
                reward_samples = []
            else:
                # Keep the emote's mood (resolve "mirror" against the live label).
                controller.set_mood(cur_emote.mood or EMOTION_TO_MOOD.get(label, Mood.NEUTRAL))

            state = controller.update(dt)

            # --- present three windows (per-eye blink enables winking) ---
            left_state = replace(state, blink=controller.blink_left)
            right_state = replace(state, blink=controller.blink_right)
            cv2.imshow("Left Eye", eye_to_bgr(renderer.render(left_state, cfg.hw.eye0.eye_side), args.eye_scale))
            cv2.imshow("Right Eye", eye_to_bgr(renderer.render(right_state, cfg.hw.eye1.eye_side), args.eye_scale))
            q_val, q_n = policy.value(arm_identity, arm_context, cur_emote.name)
            tag = "explore" if explored else f"best Q={q_val:+.2f} n={q_n}"
            draw_preview(frame, pts, label, conf, controller.mood, fps, who=who,
                         emote_text=f"{cur_emote.name} [{tag}]")
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                controller.blink()
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_learn:
            policy.save()
            print(f"Saved learned emote preferences -> {args.policy_file}")
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
