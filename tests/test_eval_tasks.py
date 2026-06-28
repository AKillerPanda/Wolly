"""Tests for the pure helpers in eval_emotion_tasks.py (no camera/model needed)."""
import numpy as np

from affect_pi.eval_emotion_tasks import (
    gather_test_samples,
    print_confusion_matrix,
    save_cm_png,
)


def _make_dataset(tmp_path, per_class=5):
    for cls in ("Happy", "Sad"):
        d = tmp_path / cls
        d.mkdir()
        for i in range(per_class):
            (d / f"{i}.jpg").write_bytes(b"x")  # content irrelevant; gather lists files
    return tmp_path


def test_gather_test_samples_skips_and_takes(tmp_path):
    root = _make_dataset(tmp_path, per_class=5)
    samples = gather_test_samples(root, skip=2, take=2)
    # 2 classes x 2 taken (images 2 and 3 after skipping 0,1)
    assert len(samples) == 4
    labels = sorted({s.label for s in samples})
    assert labels == ["Happy", "Sad"]
    names = sorted(s.image_path.name for s in samples if s.label == "Happy")
    assert names == ["2.jpg", "3.jpg"]


def test_gather_test_samples_empty_when_skip_exceeds(tmp_path):
    root = _make_dataset(tmp_path, per_class=3)
    assert gather_test_samples(root, skip=10, take=5) == []


def test_save_cm_png_writes_file(tmp_path):
    cm = np.array([[5, 1], [2, 4]], dtype=int)
    out = tmp_path / "cm.png"
    ok = save_cm_png(cm, ["Happy", "Sad"], out)
    assert ok is True            # matplotlib is a declared dependency
    assert out.exists() and out.stat().st_size > 0


def test_print_confusion_matrix_smoke(capsys):
    cm = np.array([[3, 0], [1, 2]], dtype=int)
    print_confusion_matrix(cm, ["Happy", "Sad"])
    out = capsys.readouterr().out
    assert "Confusion matrix" in out
    assert "Happy" in out
