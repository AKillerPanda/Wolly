"""Tests for the integrated live app glue (scripts/live_emotion_eyes.py)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# A picklable (module-level) stand-in for the trained classifier.
class DummyProbaModel:
    classes_ = np.array(["Angry", "Fear", "Happy", "Sad", "Suprise"])

    def predict_proba(self, x):
        # Always "Happy"-leaning probabilities, independent of input.
        return np.tile([0.10, 0.10, 0.50, 0.20, 0.10], (len(x), 1)).astype(np.float32)


def _load_app():
    if "live_app" in sys.modules:
        return sys.modules["live_app"]
    spec = importlib.util.spec_from_file_location("live_app", ROOT / "scripts" / "live_emotion_eyes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["live_app"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeController:
    def __init__(self):
        self.calls = []

    def set_mood(self, m):
        self.calls.append(("mood", m))

    def blink(self):
        self.calls.append(("blink", None))

    def wink(self, side):
        self.calls.append(("wink", side))

    def look(self, dx, dy, hold=True):
        self.calls.append(("look", (dx, dy)))


def test_emotion_to_mood_covers_all_trained_classes():
    app = _load_app()
    for cls in DummyProbaModel.classes_:
        assert str(cls) in app.EMOTION_TO_MOOD


def test_emotion_reader_predict_and_happiness(tmp_path, face_pts2d):
    joblib = pytest.importorskip("joblib")
    app = _load_app()
    payload = {"model": DummyProbaModel(), "node_offsets": [0.0] * 6,
               "classes": list(DummyProbaModel.classes_)}
    path = tmp_path / "m.joblib"
    joblib.dump(payload, path)

    reader = app.EmotionReader(path)
    label, conf = reader.predict(face_pts2d)
    assert label == "Happy"
    assert conf == pytest.approx(0.5, abs=1e-5)
    # happiness = P(Happy) + 0.3*P(Suprise) - 0.5*(P(Sad)+P(Angry)+P(Fear))
    #           = 0.5 + 0.3*0.1 - 0.5*(0.2+0.1+0.1) = 0.33
    assert reader.happiness() == pytest.approx(0.33, abs=1e-5)


def test_start_emote_mirror_uses_user_label():
    app = _load_app()
    from robot_eyes.behavior import Emote
    ctl = FakeController()
    app.start_emote(ctl, Emote("mirror", None, None), "Angry")
    assert ("mood", app.Mood.ANGRY) in ctl.calls


def test_start_emote_applies_action():
    app = _load_app()
    from robot_eyes.behavior import Emote
    from robot_eyes.config import Mood
    ctl = FakeController()
    app.start_emote(ctl, Emote("wink_left", Mood.HAPPY, "wink_left"), "Sad")
    assert ("mood", Mood.HAPPY) in ctl.calls
    assert ("wink", "left") in ctl.calls


def test_eye_to_bgr_upscales_and_converts():
    app = _load_app()
    rgb = np.zeros((10, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # pure R in RGB
    out = app.eye_to_bgr(rgb, scale=3)
    assert out.shape == (30, 24, 3)
    # after RGB->BGR the red channel is index 2
    assert out[0, 0, 2] == 200 and out[0, 0, 0] == 0
