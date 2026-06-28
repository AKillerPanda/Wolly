"""Tests for the pure dataset-conversion helpers in train_yolo_face.py."""
import numpy as np

from affect_pi.train_yolo_face import (
    FACE_CLASS_ID,
    box_to_yolo_line,
    read_boxes_by_image,
)


def test_box_to_yolo_line_normalizes_center_and_size():
    line = box_to_yolo_line((10, 20, 110, 220), w=200, h=400)
    cls, cx, cy, bw, bh = line.split()
    assert int(cls) == FACE_CLASS_ID
    assert np.isclose(float(cx), 60 / 200)     # (10+110)/2 / 200
    assert np.isclose(float(cy), 120 / 400)    # (20+220)/2 / 400
    assert np.isclose(float(bw), 100 / 200)
    assert np.isclose(float(bh), 200 / 400)


def test_box_to_yolo_line_degenerate_returns_none():
    assert box_to_yolo_line((50, 50, 50, 80), w=100, h=100) is None   # zero width


def test_box_to_yolo_line_clamps_out_of_bounds():
    # x beyond the image gets clamped to the width before normalizing.
    line = box_to_yolo_line((-10, -10, 300, 300), w=100, h=100)
    _, cx, cy, bw, bh = line.split()
    assert 0.0 <= float(cx) <= 1.0 and 0.0 <= float(cy) <= 1.0
    assert float(bw) <= 1.0 and float(bh) <= 1.0


def test_read_boxes_by_image_groups_and_drops_degenerate(tmp_path):
    csv = tmp_path / "faces.csv"
    csv.write_text(
        "image_name,width,height,x0,y0,x1,y1\n"
        "a.jpg,100,100,10,10,50,50\n"
        "a.jpg,100,100,60,60,90,90\n"
        "b.jpg,100,100,5,5,5,40\n"      # degenerate (x1<=x0) -> dropped
        "b.jpg,100,100,5,5,40,40\n",
        encoding="utf-8",
    )
    by_image = read_boxes_by_image(csv)
    assert set(by_image.keys()) == {"a.jpg", "b.jpg"}
    w, h, boxes_a = by_image["a.jpg"]
    assert (w, h) == (100, 100)
    assert len(boxes_a) == 2
    _, _, boxes_b = by_image["b.jpg"]
    assert len(boxes_b) == 1   # the degenerate row was skipped
