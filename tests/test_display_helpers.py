"""Pure helpers from the display stack that need no pygame/SPI hardware."""
import numpy as np

from robot_eyes.backends import _dirty_box, _rotate
from robot_eyes.st7789 import rgb_to_565_be
from robot_eyes import main as eyes_main


def test_rotate_identity_and_90():
    fb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(_rotate(fb, 0), fb)
    r = _rotate(fb, 90)
    assert r.shape == (3, 2, 3)   # H and W swap


def test_dirty_box_full_frame_when_no_prev():
    cur = np.zeros((10, 8, 3), dtype=np.uint8)
    assert _dirty_box(cur, None) == (0, 0, 7, 9)


def test_dirty_box_none_when_unchanged():
    cur = np.zeros((10, 8, 3), dtype=np.uint8)
    assert _dirty_box(cur, cur.copy()) is None


def test_dirty_box_bounds_of_change():
    prev = np.zeros((10, 8, 3), dtype=np.uint8)
    cur = prev.copy()
    cur[3:5, 2:6] = 255
    box = _dirty_box(cur, prev)
    assert box == (2, 3, 5, 4)   # (x0, y0, x1, y1) inclusive


def test_rgb_to_565_be_byte_layout():
    fb = np.zeros((1, 2, 3), dtype=np.uint8)
    fb[0, 0] = (255, 0, 0)     # red   -> 0xF800 big-endian -> F8 00
    fb[0, 1] = (0, 0, 255)     # blue  -> 0x001F          -> 00 1F
    out = rgb_to_565_be(fb)
    assert out == bytes([0xF8, 0x00, 0x00, 0x1F])


def test_is_pi5_false_off_pi():
    # On the dev machine /proc/device-tree/model does not exist -> False.
    assert eyes_main.is_pi5() in (True, False)
    assert eyes_main.is_pi5() is False
