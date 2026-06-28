"""Pipeline tests with fake detectors (no MediaPipe / camera needed)."""
import numpy as np

from affect_pi.model import NeutralPlaceholderModel
from affect_pi.pipeline import FrameResult, VisionPipeline
from affect_pi.status import VisionStatus
from affect_pi.trend import GaussianTrendLayer


class FakeFace:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.closed = False

    def detect(self, frame):
        return self.landmarks

    def close(self):
        self.closed = True


class FakePose:
    def __init__(self, parts):
        self.parts = parts
        self.closed = False

    def detect(self, frame):
        return self.parts

    def close(self):
        self.closed = True


def _frame():
    return np.zeros((48, 48, 3), dtype=np.uint8)


def test_status_enum_values():
    assert VisionStatus.FACE_VISIBLE.value == "FACE_VISIBLE"
    assert VisionStatus("BODY_VISIBLE") is VisionStatus.BODY_VISIBLE


def test_face_visible_path(face_pts3d):
    pipe = VisionPipeline(FakeFace(face_pts3d), FakePose(None),
                          NeutralPlaceholderModel(), GaussianTrendLayer())
    res = pipe.process(_frame())
    assert isinstance(res, FrameResult)
    assert res.status == VisionStatus.FACE_VISIBLE
    assert res.emotion.label == "neutral"
    assert res.facial_distortions is not None
    assert res.meta["n_face_landmarks"] == 478
    assert res.meta["feature_dim"] == res.facial_distortions.feature_vector().size


def test_body_visible_when_no_face():
    parts = {"NOSE": (1.0, 2.0, 0.0, 0.9)}
    pipe = VisionPipeline(FakeFace(None), FakePose(parts),
                          NeutralPlaceholderModel(), GaussianTrendLayer())
    res = pipe.process(_frame())
    assert res.status == VisionStatus.BODY_VISIBLE
    assert res.body_parts == parts
    assert res.meta["n_body_parts"] == 1


def test_cant_see_when_nothing():
    pipe = VisionPipeline(FakeFace(None), FakePose(None),
                          NeutralPlaceholderModel(), GaussianTrendLayer())
    res = pipe.process(_frame())
    assert res.status == VisionStatus.CANT_SEE
    assert res.emotion is None


def test_close_propagates_to_detectors(face_pts3d):
    face, pose = FakeFace(face_pts3d), FakePose(None)
    VisionPipeline(face, pose, NeutralPlaceholderModel(), GaussianTrendLayer()).close()
    assert face.closed and pose.closed
