"""
config.py - All tunable parameters in one place.

References for hardware constants:
  - Panel: Waveshare 1.47" LCD, 172(H)x320(V), ST7789V3, 4-wire SPI, RGB565.
    https://www.waveshare.com/wiki/1.47inch_LCD_Module
  - ST7789V3 controller RAM is 240x320; the 172-wide panel is offset by
    (240-172)/2 = 34 columns in portrait. This is COL_OFFSET below.
  - Pi 5 GPIO: use gpiozero/lgpio (NOT RPi.GPIO). SPI via spidev.
    https://gpiozero.readthedocs.io / Raspberry Pi GPIO white paper RP-006553-WP.

Coordinate convention: framebuffers are numpy arrays of shape (H, W, 3),
uint8 RGB, indexed [row(y), col(x), channel]. Portrait by default: W=172, H=320.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class Mood(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    TIRED = "tired"
    SURPRISED = "surprised"
    SAD = "sad"
    FEAR = "fear"


@dataclass(frozen=True)
class ScreenConfig:
    # Native panel resolution in PORTRAIT. The Waveshare 1.47" is 172x320.
    width: int = 172
    height: int = 320
    # Pixel block size for the "pixelated" look. Logical grid = (height//px, width//px).
    # 172 = 4*43, 320 = 4*80 -> px=4 divides both exactly (recommended).
    pixel_size: int = 4


@dataclass(frozen=True)
class EyeStyle:
    bg_color: Tuple[int, int, int] = (0, 0, 0)          # background (display off-pixels)
    eye_color: Tuple[int, int, int] = (0, 200, 255)     # robot cyan
    # Eye geometry as a fraction of the LOGICAL grid (post-pixelation grid).
    eye_w_frac: float = 0.62      # eye width  / grid width
    eye_h_frac: float = 0.46      # eye height / grid height
    corner_radius_frac: float = 0.45   # corner radius / min(eye_w, eye_h) half-extent
    # Maximum gaze travel (saccade range) as a fraction of free space.
    max_move_w_frac: float = 0.16
    max_move_h_frac: float = 0.14


@dataclass(frozen=True)
class AnimationConfig:
    target_fps: int = 30
    # Blink: time between blinks drawn uniformly from [min, max] seconds.
    blink_interval_s: Tuple[float, float] = (2.4, 6.0)
    blink_duration_s: float = 0.16     # full close+open
    min_open_frac: float = 0.06        # eye height multiplier at full blink (never 0)
    # Idle micro-saccades: retarget every [min, max] seconds, ease speed in 1/s.
    saccade_interval_s: Tuple[float, float] = (0.9, 2.8)
    saccade_radius_frac: float = 0.55  # fraction of max gaze used for idle drift
    ease_speed: float = 8.0            # higher = snappier eye movement


@dataclass(frozen=True)
class HardwareConfig:
    """Per-eye SPI + control-pin wiring for the Raspberry Pi 5.

    Default assumes BOTH panels on SPI0 using the two hardware chip-selects
    (CE0=GPIO8, CE1=GPIO7), with shared DC and RST, and one BL pin each.
    spidev addresses these as (bus=0, device=0) and (bus=0, device=1).

    If you instead wired each panel to its own bus (SPI0 + SPI1) to roughly
    double throughput, set eye0.spi_bus/spi_dev and eye1.spi_bus/spi_dev and
    give each its own DC/RST.
    """
    @dataclass(frozen=True)
    class Panel:
        spi_bus: int
        spi_dev: int          # chip-select index for that bus
        dc_pin: int           # data/command (BCM)
        rst_pin: int          # reset (BCM); set -1 if tied to Pi reset / not wired
        bl_pin: int           # backlight (BCM); set -1 if tied high
        eye_side: str         # "left" or "right" -> mirrors asymmetric moods

    spi_hz: int = 40_000_000  # 40 MHz: reliable on PH2.0 jumper leads. Bump to 60-80M
                              # MHz on short, clean wiring (ST7789 write clock max ~62.5 MHz).
    rotation: int = 0         # 0/90/180/270, applied to map portrait buffer -> panel.
    invert: bool = True       # ST7789 IPS panels normally need display inversion ON.
    bgr: bool = False         # set True if red/blue come out swapped (MADCTL RGB/BGR bit).
    # Panel-vs-RAM offsets for portrait (172 wide inside 240-wide RAM).
    col_offset: int = 34      # (240-172)/2  -- VERIFY with --calibrate.
    row_offset: int = 0

    eye0: "HardwareConfig.Panel" = field(
        default_factory=lambda: HardwareConfig.Panel(
            spi_bus=0, spi_dev=0, dc_pin=25, rst_pin=27, bl_pin=18, eye_side="left"))
    eye1: "HardwareConfig.Panel" = field(
        default_factory=lambda: HardwareConfig.Panel(
            spi_bus=0, spi_dev=1, dc_pin=25, rst_pin=27, bl_pin=24, eye_side="right"))


@dataclass(frozen=True)
class Config:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    style: EyeStyle = field(default_factory=EyeStyle)
    anim: AnimationConfig = field(default_factory=AnimationConfig)
    hw: HardwareConfig = field(default_factory=HardwareConfig)

    # Derived logical grid dimensions (post-pixelation).
    @property
    def grid_w(self) -> int:
        return self.screen.width // self.screen.pixel_size

    @property
    def grid_h(self) -> int:
        return self.screen.height // self.screen.pixel_size


DEFAULT = Config()
