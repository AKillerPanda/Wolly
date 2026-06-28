import numpy as np

from robot_eyes.config import Config, Mood
from robot_eyes.renderer import EyeRenderer, RenderState


def _fg_count(ren, cfg, state, side="left"):
    img = ren.render(state, side)
    return int((img.reshape(-1, 3) == np.array(cfg.style.eye_color)).all(axis=1).sum())


def test_render_shape_and_dtype():
    cfg = Config()
    ren = EyeRenderer(cfg)
    img = ren.render(RenderState(mood=Mood.NEUTRAL), "left")
    assert img.shape == (cfg.screen.height, cfg.screen.width, 3)
    assert img.dtype == np.uint8


def test_neutral_eye_has_foreground_pixels():
    cfg = Config()
    ren = EyeRenderer(cfg)
    assert _fg_count(ren, cfg, RenderState(mood=Mood.NEUTRAL)) > 0


def test_blink_closes_eye():
    cfg = Config()
    ren = EyeRenderer(cfg)
    open_px = _fg_count(ren, cfg, RenderState(mood=Mood.NEUTRAL, blink=1.0))
    shut_px = _fg_count(ren, cfg, RenderState(mood=Mood.NEUTRAL, blink=0.0))
    assert shut_px < open_px * 0.5


def test_surprised_is_bigger_than_neutral():
    cfg = Config()
    ren = EyeRenderer(cfg)
    neutral = _fg_count(ren, cfg, RenderState(mood=Mood.NEUTRAL))
    surprised = _fg_count(ren, cfg, RenderState(mood=Mood.SURPRISED))
    assert surprised > neutral


def test_each_mood_is_visually_distinct():
    cfg = Config()
    ren = EyeRenderer(cfg)
    counts = {m: _fg_count(ren, cfg, RenderState(mood=m)) for m in Mood}
    # The new SAD and FEAR moods must differ from neutral and from each other,
    # and SAD must differ from the visually-similar TIRED.
    assert counts[Mood.SAD] != counts[Mood.NEUTRAL]
    assert counts[Mood.FEAR] != counts[Mood.NEUTRAL]
    assert counts[Mood.SAD] != counts[Mood.TIRED]
    assert counts[Mood.SAD] != counts[Mood.FEAR]


def test_angry_lid_is_asymmetric_between_eyes():
    cfg = Config()
    ren = EyeRenderer(cfg)
    left = _fg_count(ren, cfg, RenderState(mood=Mood.ANGRY), side="left")
    right = _fg_count(ren, cfg, RenderState(mood=Mood.ANGRY), side="right")
    # angry lid drops on the inner side, which mirrors per eye -> same area but
    # the masks differ; assert the rendered eyes are not identical arrays.
    li = ren.render(RenderState(mood=Mood.ANGRY), "left")
    ri = ren.render(RenderState(mood=Mood.ANGRY), "right")
    assert not np.array_equal(li, ri)


def test_gaze_shifts_eye_center():
    cfg = Config()
    ren = EyeRenderer(cfg)
    center = ren.render(RenderState(mood=Mood.NEUTRAL, gaze_x=0.0), "left")
    shifted = ren.render(RenderState(mood=Mood.NEUTRAL, gaze_x=5.0), "left")
    assert not np.array_equal(center, shifted)
