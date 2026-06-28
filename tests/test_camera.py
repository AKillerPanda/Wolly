import numpy as np
import pytest

import affect_pi.camera as camera_mod
from affect_pi.camera import WebcamCamera, make_camera


class FakeCap:
    def __init__(self, *a, **k):
        self._open = True

    def set(self, *a):
        pass

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self._open = False


def test_make_camera_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_camera("nonsense", 0, 640, 480)


def test_make_camera_picamera2_without_lib_raises():
    # picamera2 is not installed on the dev machine -> RuntimeError with guidance.
    with pytest.raises(RuntimeError):
        make_camera("picamera2", 0, 640, 480)


def test_webcam_camera_reads(monkeypatch):
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", FakeCap)
    cam = make_camera("webcam", 0, 320, 240)
    assert isinstance(cam, WebcamCamera)
    ok, frame = cam.read()
    assert ok and frame.shape == (4, 4, 3)
    cam.release()


def test_webcam_camera_raises_when_not_opened(monkeypatch):
    class ClosedCap(FakeCap):
        def isOpened(self):
            return False

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", ClosedCap)
    with pytest.raises(RuntimeError):
        WebcamCamera(camera_index=0)
