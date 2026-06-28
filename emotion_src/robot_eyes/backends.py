"""
backends.py - Interchangeable output backends.

Two implementations of the same tiny interface:
  - SpiBackend     : drives two physical ST7789 panels on the Pi 5.
  - SimBackend     : draws both eyes side-by-side in a pygame window on a PC.

main.py picks one at runtime; the renderer/controller are identical for both.
Heavy/platform-specific imports (spidev, gpiozero, pygame) are done lazily inside
the chosen backend so the wrong one is never loaded.

Backend interface:
    .events() -> list[str]              # control tokens, e.g. "mood:happy", "quit"
    .show(left_fb, right_fb) -> None    # present one frame (two (H,W,3) uint8 arrays)
    .close() -> None
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from .config import Config, Mood


class Backend(ABC):
    @abstractmethod
    def events(self) -> List[str]: ...
    @abstractmethod
    def show(self, left_fb: np.ndarray, right_fb: np.ndarray) -> None: ...
    @abstractmethod
    def close(self) -> None: ...


def _rotate(fb: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return fb
    return np.rot90(fb, k=(rotation // 90)).copy()


def _dirty_box(cur: np.ndarray, prev: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box (x0,y0,x1,y1 inclusive) of pixels that changed vs prev.
    Returns None if nothing changed; full frame if prev is None."""
    if prev is None:
        return (0, 0, cur.shape[1] - 1, cur.shape[0] - 1)
    diff = np.any(cur != prev, axis=2)
    if not diff.any():
        return None
    rows = np.where(diff.any(axis=1))[0]
    cols = np.where(diff.any(axis=0))[0]
    return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))


# --------------------------------------------------------------------------- #
#  Hardware: two ST7789 panels                                                #
# --------------------------------------------------------------------------- #
class SpiBackend(Backend):
    def __init__(self, cfg: Config, use_dirty_rect: bool = True):
        from gpiozero import DigitalOutputDevice

        from .st7789 import ST7789
        self.cfg = cfg
        self.rotation = cfg.hw.rotation
        self.use_dirty = use_dirty_rect

        # One DigitalOutputDevice per UNIQUE pin, shared across panels. The two
        # panels here share DC and RST (only their CS and backlight differ); a
        # naive per-panel allocation would raise gpiozero's GPIOPinInUse on the
        # second panel. -1 means "not wired" (e.g. backlight tied high).
        self._gpio: dict[int, object] = {}

        def shared(pin: int):
            if pin is None or pin < 0:
                return None
            if pin not in self._gpio:
                self._gpio[pin] = DigitalOutputDevice(pin)
            return self._gpio[pin]

        reset_done: set[int] = set()

        def make(panel):
            rst_dev = shared(panel.rst_pin)
            # Only the first panel on a given RST line pulses it; a second pulse
            # would reset the already-initialised panel.
            do_hw_reset = rst_dev is not None and panel.rst_pin not in reset_done
            reset_done.add(panel.rst_pin)
            return ST7789(
                spi_bus=panel.spi_bus, spi_dev=panel.spi_dev,
                dc_pin=shared(panel.dc_pin), rst_pin=rst_dev, bl_pin=shared(panel.bl_pin),
                width=cfg.screen.width, height=cfg.screen.height,
                rotation=cfg.hw.rotation, invert=cfg.hw.invert, bgr=cfg.hw.bgr,
                col_offset=cfg.hw.col_offset, row_offset=cfg.hw.row_offset,
                spi_hz=cfg.hw.spi_hz, do_hw_reset=do_hw_reset)

        self._panels = [make(cfg.hw.eye0), make(cfg.hw.eye1)]
        self._prev: list[Optional[np.ndarray]] = [None, None]

    def events(self) -> List[str]:
        return []   # headless on the Pi; mood changes come from the demo schedule

    def show(self, left_fb, right_fb) -> None:
        for i, fb in enumerate((left_fb, right_fb)):
            out = _rotate(fb, self.rotation)
            if self.use_dirty:
                box = _dirty_box(out, self._prev[i])
                if box is not None:
                    self._panels[i].blit(out, box)
                self._prev[i] = out
            else:
                self._panels[i].blit(out)

    def close(self) -> None:
        for p in self._panels:
            try:
                p.close()
            except Exception:
                pass
        # Release the shared DC/RST/BL devices the backend owns.
        for dev in self._gpio.values():
            try:
                dev.close()
            except Exception:
                pass
        self._gpio.clear()


# --------------------------------------------------------------------------- #
#  Simulator: pygame window                                                   #
# --------------------------------------------------------------------------- #
class SimBackend(Backend):
    _KEY_MOODS = {"1": Mood.NEUTRAL, "2": Mood.HAPPY, "3": Mood.ANGRY,
                  "4": Mood.TIRED, "5": Mood.SURPRISED}

    def __init__(self, cfg: Config, zoom: int = 2, gap: int = 24):
        import pygame
        self._pg = pygame
        self.cfg = cfg
        self.rotation = cfg.hw.rotation

        # Determine on-screen panel size after rotation.
        if cfg.hw.rotation in (90, 270):
            self._pw, self._ph = cfg.screen.height, cfg.screen.width
        else:
            self._pw, self._ph = cfg.screen.width, cfg.screen.height
        self.zoom = zoom
        self.gap = gap

        pygame.init()
        w = self._pw * zoom * 2 + gap
        h = self._ph * zoom
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Robot Eyes (simulator)  keys: 1-5 mood  arrows look  space blink  q quit")
        self._bg = pygame.Color(20, 20, 22)

    def events(self) -> List[str]:
        pg = self._pg
        out: List[str] = []
        for e in pg.event.get():
            if e.type == pg.QUIT:
                out.append("quit")
            elif e.type == pg.KEYDOWN:
                name = pg.key.name(e.key)
                if name in ("q", "escape"):
                    out.append("quit")
                elif name in self._KEY_MOODS:
                    out.append(f"mood:{self._KEY_MOODS[name].value}")
                elif name == "space":
                    out.append("blink")
                elif name == "left":
                    out.append("look:-1,0")
                elif name == "right":
                    out.append("look:1,0")
                elif name == "up":
                    out.append("look:0,-1")
                elif name == "down":
                    out.append("look:0,1")
                elif name == "c":
                    out.append("look:0,0")
        return out

    def _blit_eye(self, fb: np.ndarray, x_offset: int) -> None:
        pg = self._pg
        out = _rotate(fb, self.rotation)
        # pygame surfarray expects (W, H, 3); our fb is (H, W, 3).
        surf = pg.surfarray.make_surface(np.transpose(out, (1, 0, 2)))
        surf = pg.transform.scale(surf, (self._pw * self.zoom, self._ph * self.zoom))
        self._screen.blit(surf, (x_offset, 0))

    def show(self, left_fb, right_fb) -> None:
        self._screen.fill(self._bg)
        self._blit_eye(left_fb, 0)
        self._blit_eye(right_fb, self._pw * self.zoom + self.gap)
        self._pg.display.flip()

    def close(self) -> None:
        self._pg.quit()
