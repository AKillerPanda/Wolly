#!/usr/bin/env python3
"""
run_on_pi.py - Affect-Pi deploy entry point.

Camera -> face landmarks -> emotion -> identity -> adaptive emote -> robot eyes.
The SAME renderer drives either the two physical ST7789 SPI panels (default on
the Pi) or on-screen windows (for testing on a laptop / a Pi with a monitor).

    python3 run_on_pi.py                 # drive the two LCD panels (Raspberry Pi)
    python3 run_on_pi.py --display windows   # show OpenCV windows instead (dev)
    python3 run_on_pi.py --display both      # panels + a camera preview window
    python3 run_on_pi.py --calibrate         # panel alignment test pattern, then quit

Everything is resolved relative to this file, so the whole emotion_src/ folder is
drag-and-drop: copy it anywhere on the Pi and run this script from inside it.

Controls (only when a preview window is shown): q = quit, space = blink.
Headless (panels only): Ctrl-C to quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

# Make the bundled packages importable no matter where the folder is dropped.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from affect_pi.mdistortion_tasks import detect_video, feature_from_pts, load_landmarker  # noqa: E402
from affect_pi.face_identity import FaceRegistry, IdentityTracker  # noqa: E402
from robot_eyes.behavior import EMOTES, AdaptivePolicy, Emote  # noqa: E402
from robot_eyes.config import Config, Mood  # noqa: E402
from robot_eyes.controller import EyeController  # noqa: E402
from robot_eyes.renderer import EyeRenderer  # noqa: E402

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

LANDMARKER_MODEL = ROOT / "models" / "face_landmarker.task"
EMOTION_MODEL = ROOT / "artifacts" / "emotion_tasks_model.joblib"
FACES_FILE = ROOT / "artifacts" / "known_faces.txt"
POLICY_FILE = ROOT / "artifacts" / "emote_policy.txt"

EMOTION_TO_MOOD = {
    "Happy": Mood.HAPPY, "Angry": Mood.ANGRY, "Sad": Mood.SAD,
    "Fear": Mood.FEAR, "Suprise": Mood.SURPRISED, "Surprise": Mood.SURPRISED,
}


# --------------------------------------------------------------------------- #
#  Emotion classifier (smoothed) + happiness reward                           #
# --------------------------------------------------------------------------- #
class EmotionReader:
    def __init__(self, model_path: Path, smooth: int = 7):
        if joblib is None:
            raise SystemExit("joblib is required (pip install joblib).")
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.offsets = np.asarray(payload.get("node_offsets", [0.0] * 6), dtype=np.float32)
        self.classes = list(getattr(self.model, "classes_", payload.get("classes", [])))
        self._idx = {str(c): i for i, c in enumerate(self.classes)}
        self.history: deque = deque(maxlen=smooth)
        self.last_mean = None

    def predict(self, pts):
        feat = feature_from_pts(pts, self.offsets)
        if feat is None:
            return None, 0.0
        probs = np.asarray(self.model.predict_proba(feat.reshape(1, -1))[0], dtype=np.float32)
        self.history.append(probs)
        mean = np.mean(np.stack(self.history, axis=0), axis=0)
        self.last_mean = mean
        i = int(np.argmax(mean))
        return str(self.classes[i]), float(mean[i])

    def happiness(self) -> float:
        if self.last_mean is None:
            return 0.0
        def p(name):
            j = self._idx.get(name)
            return float(self.last_mean[j]) if j is not None else 0.0
        return p("Happy") + 0.3 * p("Suprise") - 0.5 * (p("Sad") + p("Angry") + p("Fear"))


def start_emote(controller: EyeController, emote: Emote, user_label) -> None:
    controller.set_mood(emote.mood or EMOTION_TO_MOOD.get(user_label, Mood.NEUTRAL))
    if emote.action == "blink":
        controller.blink()
    elif emote.action == "wink_left":
        controller.wink("left")
    elif emote.action == "wink_right":
        controller.wink("right")
    elif emote.action == "look":
        controller.look(emote.look[0], emote.look[1], hold=False)


# --------------------------------------------------------------------------- #
#  Camera: Pi Camera 2 (CSI) preferred, USB webcam fallback                    #
# --------------------------------------------------------------------------- #
class Camera:
    def __init__(self, index: int, width: int = 640, height: int = 480):
        self.kind = None
        self._cam = None
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.configure(cam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}))
            cam.start()
            time.sleep(0.5)
            self.kind, self._cam = "picamera2", cam
            print("Camera: Pi Camera 2 (CSI).")
            return
        except Exception as exc:
            print(f"Pi Camera 2 unavailable ({exc.__class__.__name__}); trying USB webcam...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.kind, self._cam = "cv2", cap
            print(f"Camera: USB webcam index {index}.")
            return
        raise SystemExit("No camera found (neither Pi Camera 2 nor USB webcam).")

    def read(self):
        if self.kind == "picamera2":
            rgb = self._cam.capture_array()
            if rgb is None:
                return False, None
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self._cam.read()

    def release(self):
        try:
            if self.kind == "picamera2":
                self._cam.stop()
            else:
                self._cam.release()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  Display: SPI panels and/or OpenCV windows                                  #
# --------------------------------------------------------------------------- #
class PanelDisplay:
    def __init__(self, cfg: Config):
        from robot_eyes.backends import SpiBackend
        self._be = SpiBackend(cfg)

    def show(self, left_rgb, right_rgb):
        self._be.show(left_rgb, right_rgb)   # st7789 expects RGB framebuffers

    def close(self):
        self._be.close()


class WindowDisplay:
    def __init__(self, cfg: Config, scale: int = 2):
        self.scale = scale
        self.w = cfg.screen.width * scale
        for i, name in enumerate(("Left Eye", "Right Eye")):
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(name, 40 + i * (self.w + 20), 80)

    def show(self, left_rgb, right_rgb):
        for name, fb in (("Left Eye", left_rgb), ("Right Eye", right_rgb)):
            bgr = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (fb.shape[1] * self.scale, fb.shape[0] * self.scale),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imshow(name, bgr)

    def close(self):
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Affect-Pi: camera -> emotion -> robot eyes (Pi deploy).")
    p.add_argument("--display", choices=["panels", "windows", "both"], default="panels")
    p.add_argument("--camera", type=int, default=0, help="USB webcam index (ignored for CSI)")
    p.add_argument("--mirror", action="store_true", help="Mirror the camera horizontally")
    p.add_argument("--min-conf", type=float, default=0.35,
                   help="Below this smoothed confidence the eyes idle neutral")
    p.add_argument("--epsilon", type=float, default=0.2, help="Emote exploration rate")
    p.add_argument("--no-learn", action="store_true", help="Use learned preferences but stop updating")
    p.add_argument("--calibrate", action="store_true", help="Draw the panel alignment pattern and quit")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config()

    if args.calibrate:
        from robot_eyes.main import calibrate
        calibrate(cfg)
        return 0

    for path, what in [(LANDMARKER_MODEL, "FaceLandmarker model"), (EMOTION_MODEL, "emotion model")]:
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}")

    landmarker = load_landmarker(LANDMARKER_MODEL, video=True)
    reader = EmotionReader(EMOTION_MODEL)
    renderer = EyeRenderer(cfg)
    controller = EyeController(cfg, seed=args.seed)
    registry = FaceRegistry.load(FACES_FILE)
    identity = IdentityTracker(registry, auto_enroll=True)
    policy = AdaptivePolicy(EMOTES, epsilon=args.epsilon, seed=args.seed,
                            path=None if args.no_learn else POLICY_FILE)
    camera = Camera(args.camera)

    displays = []
    if args.display in ("panels", "both"):
        displays.append(PanelDisplay(cfg))
    if args.display in ("windows", "both"):
        displays.append(WindowDisplay(cfg))
    show_window = args.display in ("windows", "both")
    print(f"Running. display={args.display}. Emotions={reader.classes}. "
          f"{'q to quit' if show_window else 'Ctrl-C to quit'}.")

    start = time.time()
    last = time.perf_counter()
    last_ts = -1
    target_dt = 1.0 / cfg.anim.target_fps

    cur_emote = None
    emote_elapsed = 0.0
    rewards: list[float] = []
    arm_id, arm_ctx = "anon", "neutral"

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera read failed.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - last
            last = now

            ts = max(int((time.time() - start) * 1000), last_ts + 1)
            last_ts = ts
            pts = detect_video(landmarker, frame, ts)

            identity.update(pts)
            label, conf = (None, 0.0)
            if pts is not None:
                label, conf = reader.predict(pts)
                rewards.append(reader.happiness())
            emote_elapsed += dt

            if cur_emote is None or emote_elapsed >= cur_emote.duration:
                if cur_emote is not None and rewards and not args.no_learn:
                    policy.update(arm_id, arm_ctx, cur_emote, float(np.mean(rewards)))
                arm_id = identity.last_label or "anon"
                arm_ctx = label if (label is not None and conf >= args.min_conf) else "neutral"
                cur_emote, _ = policy.select(arm_id, arm_ctx)
                start_emote(controller, cur_emote, label)
                emote_elapsed, rewards = 0.0, []
            else:
                controller.set_mood(cur_emote.mood or EMOTION_TO_MOOD.get(label, Mood.NEUTRAL))

            state = controller.update(dt)
            left = renderer.render(replace(state, blink=controller.blink_left), cfg.hw.eye0.eye_side)
            right = renderer.render(replace(state, blink=controller.blink_right), cfg.hw.eye1.eye_side)
            for d in displays:
                d.show(left, right)

            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    controller.blink()

            sleep = target_dt - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_learn:
            policy.save()
        for d in displays:
            d.close()
        camera.release()
        landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
