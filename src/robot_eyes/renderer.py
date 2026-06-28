"""
renderer.py - Pure, hardware-free eye rendering with numpy.

Renders ONE eye into an (H, W, 3) uint8 RGB framebuffer. The renderer is
deterministic given a RenderState, which makes it unit-testable and lets the
same pixels go to either the SPI panel or the desktop simulator.

Design:
  1. Draw the eye on a small logical grid (grid_h x grid_w) using signed-distance
     masks (crisp, anti-alias-free -> intentionally blocky).
  2. Carve mood eyelids by overwriting eye pixels with the background colour.
  3. Upscale to native resolution by integer nearest-neighbour repeat -> the
     "pixelated" aesthetic, with each logical cell becoming a pixel_size block.

Rounded-rectangle mask uses the standard 2-D rounded-box SDF
(Inigo Quilez, "distance functions"): d = |length(max(|p|-b+r,0))| + min(max(...),0) - r.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import Config, Mood


@dataclass
class RenderState:
    mood: Mood = Mood.NEUTRAL
    blink: float = 1.0        # 1.0 = fully open, ->0 = closed (eye-height multiplier)
    gaze_x: float = 0.0       # gaze offset in logical grid units (+ = right)
    gaze_y: float = 0.0       # gaze offset in logical grid units (+ = down)


class EyeRenderer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        gh, gw = cfg.grid_h, cfg.grid_w
        # Cache coordinate grids (float) once; reused every frame.
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
        self._xx = xx
        self._yy = yy
        self._gh = gh
        self._gw = gw
        self._bg = np.array(cfg.style.bg_color, dtype=np.uint8)
        self._fg = np.array(cfg.style.eye_color, dtype=np.uint8)

    # ----- low-level mask helpers (all return bool arrays of shape (gh, gw)) -----

    def _rounded_rect(self, cx, cy, w, h, r) -> np.ndarray:
        hw, hh = w / 2.0, h / 2.0
        r = float(np.clip(r, 0.0, min(hw, hh)))
        ax = np.abs(self._xx - cx) - (hw - r)
        ay = np.abs(self._yy - cy) - (hh - r)
        ox = np.maximum(ax, 0.0)
        oy = np.maximum(ay, 0.0)
        d = np.sqrt(ox * ox + oy * oy) + np.minimum(np.maximum(ax, ay), 0.0) - r
        return d <= 0.0

    def _above_line(self, x0, y0, x1, y1) -> np.ndarray:
        """True where a pixel is ABOVE (smaller y than) the line through the two
        points, evaluated per-column by linear interpolation in x."""
        if x1 == x0:
            return np.zeros((self._gh, self._gw), dtype=bool)
        slope = (y1 - y0) / (x1 - x0)
        line_y = y0 + slope * (self._xx - x0)
        return self._yy <= line_y

    # ----- public API -----

    def render(self, state: RenderState, eye_side: str) -> np.ndarray:
        cfg = self.cfg
        gh, gw = self._gh, self._gw
        st = cfg.style

        eye_w = st.eye_w_frac * gw
        eye_h = st.eye_h_frac * gh * max(state.blink, cfg.anim.min_open_frac)

        cx = gw / 2.0 + state.gaze_x
        cy = gh / 2.0 + state.gaze_y

        # Surprised: open wider/taller and rounder.
        if state.mood is Mood.SURPRISED:
            eye_w *= 1.12
            eye_h *= 1.30
        elif state.mood is Mood.FEAR:
            # Fear: eyes wide and tall (tense), raised slightly up the face.
            eye_w *= 1.06
            eye_h *= 1.20
            cy -= eye_h * 0.04

        r = st.corner_radius_frac * min(eye_w, eye_h) / 2.0
        eye = self._rounded_rect(cx, cy, eye_w, eye_h, r)

        # Eyelids: overwrite part of the eye with background.
        eye = self._apply_mood_lid(eye, state.mood, eye_side, cx, cy, eye_w, eye_h)

        # Compose small RGB frame, then pixel-upscale to native resolution.
        small = np.empty((gh, gw, 3), dtype=np.uint8)
        small[:] = self._bg
        small[eye] = self._fg
        return self._upscale(small)

    def _apply_mood_lid(self, eye, mood, eye_side, cx, cy, ew, eh) -> np.ndarray:
        top = cy - eh / 2.0
        bot = cy + eh / 2.0
        left = cx - ew / 2.0
        right = cx + ew / 2.0
        # inner = side toward the centre of the face. By convention the "left"
        # panel's inner edge is its right side; the "right" panel's inner is left.
        inner_is_right = (eye_side == "left")

        if mood is Mood.HAPPY:
            # Lower eyelid rises in a curve: subtract a big circle centred below
            # the eye so the remaining bottom edge bows upward (cheerful squint).
            rc = eh * 1.15
            cyc = bot + eh * 0.30
            dist2 = (self._xx - cx) ** 2 + (self._yy - cyc) ** 2
            eye = eye & ~(dist2 <= rc * rc)

        elif mood is Mood.ANGRY:
            # Upper eyelid descends, lower on the INNER side (brows toward nose).
            drop = eh * 0.55
            if inner_is_right:
                y_left, y_right = top, top + drop
            else:
                y_left, y_right = top + drop, top
            lid = self._above_line(left, y_left, right, y_right)
            eye = eye & ~lid

        elif mood is Mood.TIRED:
            # Heavy upper eyelid, lower on the OUTER side, covering ~top 45%.
            drop_out = eh * 0.55
            drop_in = eh * 0.30
            if inner_is_right:
                y_left, y_right = top + drop_out, top + drop_in
            else:
                y_left, y_right = top + drop_in, top + drop_out
            lid = self._above_line(left, y_left, right, y_right)
            eye = eye & ~lid

        elif mood is Mood.SAD:
            # Worried/sad: upper lid lower on the OUTER side (inner corner raised)
            # -- the inverse slant of ANGRY, and lighter than TIRED.
            drop_out = eh * 0.42
            drop_in = eh * 0.05
            if inner_is_right:
                y_left, y_right = top + drop_out, top + drop_in
            else:
                y_left, y_right = top + drop_in, top + drop_out
            lid = self._above_line(left, y_left, right, y_right)
            eye = eye & ~lid

        elif mood is Mood.FEAR:
            # Tense raised upper lid: a thin flat lid clipped across the very top,
            # which together with the wide/tall eye from render() reads as alarmed.
            ylid = top + eh * 0.12
            lid = self._above_line(left, ylid, right, ylid)
            eye = eye & ~lid

        # NEUTRAL / SURPRISED: no lid.
        return eye

    def _upscale(self, small: np.ndarray) -> np.ndarray:
        ps = self.cfg.screen.pixel_size
        big = np.repeat(np.repeat(small, ps, axis=0), ps, axis=1)
        H, W = self.cfg.screen.height, self.cfg.screen.width
        # Crop/pad in case width/height are not exact multiples of pixel_size.
        big = big[:H, :W]
        if big.shape[0] != H or big.shape[1] != W:
            out = np.empty((H, W, 3), dtype=np.uint8)
            out[:] = self._bg
            out[:big.shape[0], :big.shape[1]] = big
            big = out
        return big
