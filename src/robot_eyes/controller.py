"""
controller.py - Animation state machine driving the RenderState over time.

Owns: current mood, blink timing, winks, and gaze (saccade) easing. Pure Python +
numpy RNG; no hardware. Both eyes share one controller so they look in the same
direction together (lid asymmetry is handled per-eye in the renderer).

Blinks close both eyes; a wink closes only one. After each update() the per-eye
closit­ion is exposed as ``blink_left`` / ``blink_right`` (1.0 = open, ->0 = shut)
so the caller can render each eye with its own lid; ``RenderState.blink`` carries
the more-closed of the two for single-state consumers.

Time model: caller passes elapsed dt (seconds) each update(); the controller is
frame-rate independent.
"""
from __future__ import annotations

import numpy as np

from .config import Config, Mood
from .renderer import RenderState


class EyeController:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self.mood = Mood.NEUTRAL

        a = cfg.anim
        self._blinking = False
        self._blink_t = 0.0
        self._blink_sides = ("left", "right")   # which eyes the current close affects
        self._next_blink = self._rng.uniform(*a.blink_interval_s)
        # Per-eye openness after the latest update(): 1.0 = open, ->0 = shut.
        self.blink_left = 1.0
        self.blink_right = 1.0

        # Max gaze travel in grid units.
        self._max_x = cfg.style.max_move_w_frac * cfg.grid_w
        self._max_y = cfg.style.max_move_h_frac * cfg.grid_h
        self._gaze = np.zeros(2, dtype=np.float64)        # current (x, y)
        self._gaze_target = np.zeros(2, dtype=np.float64)
        self._next_saccade = self._rng.uniform(*a.saccade_interval_s)
        self._gaze_held = False   # True after an explicit look(); disables idle drift

    # ----- external control -----

    def set_mood(self, mood: Mood) -> None:
        self.mood = mood

    def look(self, dx_frac: float, dy_frac: float, hold: bool = True) -> None:
        """Point the gaze. dx_frac/dy_frac in [-1, 1] (fraction of max travel)."""
        self._gaze_target[0] = float(np.clip(dx_frac, -1, 1)) * self._max_x
        self._gaze_target[1] = float(np.clip(dy_frac, -1, 1)) * self._max_y
        self._gaze_held = hold
        self._next_saccade = self._rng.uniform(*self.cfg.anim.saccade_interval_s)

    def blink(self) -> None:
        """Close both eyes once."""
        if not self._blinking:
            self._blinking = True
            self._blink_t = 0.0
            self._blink_sides = ("left", "right")

    def wink(self, side: str = "left") -> None:
        """Close just one eye once. side = 'left' or 'right'."""
        if not self._blinking:
            self._blinking = True
            self._blink_t = 0.0
            self._blink_sides = (side,)

    # ----- per-frame update -----

    def update(self, dt: float) -> RenderState:
        a = self.cfg.anim
        # --- blink / wink ---
        if self._blinking:
            self._blink_t += dt
            p = self._blink_t / a.blink_duration_s
            if p >= 1.0:
                self._blinking = False
                blink_factor = 1.0
                self._next_blink = self._rng.uniform(*a.blink_interval_s)
            else:
                # V-shape: 1 -> 0 -> 1 across the duration.
                blink_factor = abs(2.0 * p - 1.0)
        else:
            blink_factor = 1.0
            self._next_blink -= dt
            if self._next_blink <= 0.0:
                self.blink()

        # Apply the close only to the affected eye(s); the other stays open.
        if self._blinking:
            self.blink_left = blink_factor if "left" in self._blink_sides else 1.0
            self.blink_right = blink_factor if "right" in self._blink_sides else 1.0
        else:
            self.blink_left = self.blink_right = 1.0

        # --- idle saccades ---
        if not self._gaze_held:
            self._next_saccade -= dt
            if self._next_saccade <= 0.0:
                rad = a.saccade_radius_frac
                self._gaze_target[0] = self._rng.uniform(-rad, rad) * self._max_x
                self._gaze_target[1] = self._rng.uniform(-rad, rad) * self._max_y
                self._next_saccade = self._rng.uniform(*a.saccade_interval_s)

        # Critically-damped-ish exponential easing toward target.
        k = min(1.0, a.ease_speed * dt)
        self._gaze += (self._gaze_target - self._gaze) * k

        return RenderState(
            mood=self.mood,
            blink=min(self.blink_left, self.blink_right),
            gaze_x=float(self._gaze[0]),
            gaze_y=float(self._gaze[1]),
        )
