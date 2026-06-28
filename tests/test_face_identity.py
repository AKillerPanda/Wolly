import numpy as np
import pytest

from affect_pi.face_identity import (
    SIGNATURE_DIM,
    FaceRegistry,
    IdentityTracker,
    identity_signature,
    signature_distance,
)


def _rot(pts, deg):
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
    return (pts @ R.T).astype(np.float32)


def test_signature_dimension(face_pts2d):
    sig = identity_signature(face_pts2d)
    assert sig.shape == (SIGNATURE_DIM,) == (136,)


def test_signature_none_when_too_few_points():
    assert identity_signature(np.zeros((10, 2), dtype=np.float32)) is None
    assert identity_signature(None) is None


def test_signature_invariant_to_translation_rotation_scale(face_pts2d):
    base = identity_signature(face_pts2d)
    translated = identity_signature(face_pts2d + np.array([37.0, -12.0], dtype=np.float32))
    rotated = identity_signature(_rot(face_pts2d, 25.0))
    scaled = identity_signature(face_pts2d * 1.7)
    assert np.allclose(base, translated, atol=1e-4)
    assert np.allclose(base, rotated, atol=1e-4)
    assert np.allclose(base, scaled, atol=1e-4)


def test_same_face_closer_than_different_face(rng):
    a = rng.uniform([100, 80], [540, 400], size=(478, 2)).astype(np.float32)
    b = a.copy()
    b[:, 1] *= 1.25   # different vertical proportions = different person
    sig_a = identity_signature(a)
    sig_a2 = identity_signature(a + rng.normal(0, 0.5, a.shape).astype(np.float32))
    sig_b = identity_signature(b)
    assert signature_distance(sig_a, sig_a2) < signature_distance(sig_a, sig_b)


def test_registry_roundtrip(tmp_path, face_pts2d):
    path = tmp_path / "known.txt"
    reg = FaceRegistry.load(path)
    assert len(reg) == 0
    sig = identity_signature(face_pts2d)
    rec = reg.add(sig, "Anuska", n_samples=5)
    reg.save()

    reloaded = FaceRegistry.load(path)
    assert len(reloaded) == 1
    assert reloaded.records[0].label == "Anuska"
    assert reloaded.records[0].n_samples == 5
    assert np.allclose(reloaded.records[0].signature, rec.signature, atol=1e-4)


def test_registry_match_threshold(face_pts2d):
    reg = FaceRegistry(path=None)  # in-memory
    sig = identity_signature(face_pts2d)
    reg.add(sig, "me")
    rec, dist = reg.match(sig, threshold=0.045)
    assert rec is not None and dist == pytest.approx(0.0, abs=1e-5)
    rec2, _ = reg.match(sig + 10.0, threshold=0.045)   # absurdly far
    assert rec2 is None


def test_reinforce_updates_running_mean(face_pts2d):
    reg = FaceRegistry(path=None)
    sig = identity_signature(face_pts2d)
    rec = reg.add(sig, "me", n_samples=1)
    reg.reinforce(rec, sig + 2.0)
    assert rec.n_samples == 2
    assert np.allclose(rec.signature, sig + 1.0, atol=1e-4)   # mean of sig and sig+2


def test_identity_tracker_auto_enrolls(tmp_path, face_pts2d):
    reg = FaceRegistry.load(tmp_path / "k.txt")
    tracker = IdentityTracker(reg, match_threshold=0.045, enroll_after=5, auto_enroll=True)
    status = ""
    for _ in range(5):
        status = tracker.update(face_pts2d)
    assert "enrolled" in status
    assert len(reg) == 1
    # now recognised, not re-enrolled.
    again = tracker.update(face_pts2d)
    assert "user1" in again or tracker.last_label is not None
    assert len(reg) == 1


def test_identity_tracker_handles_no_face():
    reg = FaceRegistry(path=None)
    tracker = IdentityTracker(reg)
    assert tracker.update(None) == "no face"


def test_identity_tracker_no_autoenroll(face_pts2d):
    reg = FaceRegistry(path=None)
    tracker = IdentityTracker(reg, auto_enroll=False)
    status = tracker.update(face_pts2d)
    assert "unknown" in status
    assert len(reg) == 0
