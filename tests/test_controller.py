import numpy as np

from robot_eyes.config import Config
from robot_eyes.controller import EyeController
from robot_eyes.renderer import RenderState


def _ctrl(seed=0):
    return EyeController(Config(), seed=seed)


def test_update_returns_render_state():
    c = _ctrl()
    s = c.update(0.016)
    assert isinstance(s, RenderState)
    assert 0.0 <= s.blink <= 1.0


def test_blink_closes_both_eyes_midway():
    c = _ctrl()
    dur = c.cfg.anim.blink_duration_s
    c.blink()
    c.update(dur / 2)
    assert c.blink_left < 0.5 and c.blink_right < 0.5
    # completes and reopens after the full duration.
    c.update(dur)
    assert c.blink_left == 1.0 and c.blink_right == 1.0


def test_wink_closes_only_one_eye():
    c = _ctrl()
    dur = c.cfg.anim.blink_duration_s
    c.wink("left")
    c.update(dur / 2)
    assert c.blink_left < 0.5
    assert c.blink_right == 1.0
    # RenderState carries the more-closed value.
    s = c.update(0.0)
    assert s.blink == min(c.blink_left, c.blink_right)


def test_wink_right_side():
    c = _ctrl()
    dur = c.cfg.anim.blink_duration_s
    c.wink("right")
    c.update(dur / 2)
    assert c.blink_right < 0.5
    assert c.blink_left == 1.0


def test_look_moves_gaze_toward_target():
    c = _ctrl()
    c.look(1.0, 0.0, hold=True)
    s = None
    for _ in range(40):
        s = c.update(0.05)
    assert s.gaze_x > 0.0          # moved toward +x target
    assert abs(s.gaze_y) < abs(s.gaze_x)
    # held gaze disables idle drift, so it should settle near the max travel.
    assert s.gaze_x > 0.5 * c._max_x


def test_set_mood_is_reflected():
    from robot_eyes.config import Mood
    c = _ctrl()
    c.set_mood(Mood.HAPPY)
    assert c.update(0.016).mood is Mood.HAPPY


def test_auto_blink_eventually_triggers():
    c = _ctrl(seed=3)
    blinked = False
    for _ in range(2000):           # ~ up to 33s at 60fps -> covers max interval
        c.update(1 / 60)
        if c.blink_left < 1.0:
            blinked = True
            break
    assert blinked
