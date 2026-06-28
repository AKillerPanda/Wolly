"""
st7789.py - Minimal ST7789V3 driver over spidev, with gpiozero control pins.

Hardware-only module: imports spidev / gpiozero, so it is imported lazily by
backends.py and never loaded on a desktop simulator.

Why these choices:
  - spidev for the bus: SPI is exposed at /dev/spidevB.D on the Pi 5 and works
    unchanged (the RP1 change only affected raw GPIO register access).
  - gpiozero for DC/RST/BL: it auto-selects the lgpio backend and the correct
    gpiochip on the Pi 5, so we avoid the RPi.GPIO breakage and chip=0/4 ambiguity.
    (See Raspberry Pi GPIO white paper RP-006553-WP.)

RGB565 is sent big-endian (high byte first), which is what ST7789 expects after
COLMOD=0x55 (16 bpp).
"""
from __future__ import annotations

import time
import numpy as np

# Command set (ST7789 datasheet)
_SWRESET = 0x01
_SLPOUT = 0x11
_NORON = 0x13
_INVOFF = 0x20
_INVON = 0x21
_DISPON = 0x29
_CASET = 0x2A
_RASET = 0x2B
_RAMWR = 0x2C
_COLMOD = 0x3A
_MADCTL = 0x36

# MADCTL bits
_MADCTL_MY = 0x80
_MADCTL_MX = 0x40
_MADCTL_MV = 0x20
_MADCTL_BGR = 0x08

# rotation -> (MY,MX,MV) base value
_ROT = {0: 0x00, 90: _MADCTL_MV | _MADCTL_MX,
        180: _MADCTL_MX | _MADCTL_MY, 270: _MADCTL_MV | _MADCTL_MY}


def rgb_to_565_be(fb: np.ndarray) -> bytes:
    """(h, w, 3) uint8 RGB -> big-endian RGB565 bytes."""
    r = fb[:, :, 0].astype(np.uint16)
    g = fb[:, :, 1].astype(np.uint16)
    b = fb[:, :, 2].astype(np.uint16)
    v = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    out = np.empty((fb.shape[0], fb.shape[1], 2), dtype=np.uint8)
    out[:, :, 0] = (v >> 8).astype(np.uint8)   # high byte first (big-endian)
    out[:, :, 1] = (v & 0xFF).astype(np.uint8)
    return out.tobytes()


def _as_output(value):
    """Accept an int BCM pin, an already-built gpiozero device, or None / -1.

    Two panels sharing a control line (e.g. one DC or RST wired to both) must
    share a single ``DigitalOutputDevice`` -- allocating the same pin twice raises
    gpiozero's GPIOPinInUse. So callers driving multiple panels pass pre-built
    (shared) devices here; passing a raw int keeps single-panel use simple.
    """
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            return None
        from gpiozero import DigitalOutputDevice
        return DigitalOutputDevice(value)
    return value  # already a device (possibly shared between panels)


class ST7789:
    def __init__(self, *, spi_bus, spi_dev, dc_pin, rst_pin, bl_pin,
                 width, height, rotation, invert, bgr,
                 col_offset, row_offset, spi_hz, do_hw_reset=True):
        import spidev

        self.width = width
        self.height = height
        self.col_offset = col_offset
        self.row_offset = row_offset

        self._spi = spidev.SpiDev()
        self._spi.open(spi_bus, spi_dev)
        self._spi.max_speed_hz = spi_hz
        self._spi.mode = 0

        # dc_pin/rst_pin/bl_pin may be ints (single panel) or shared devices
        # (multi-panel, created once by the backend).
        self._dc = _as_output(dc_pin)
        self._rst = _as_output(rst_pin)
        self._bl = _as_output(bl_pin)
        # We own a device (and should close it) only if we created it from an int.
        self._owns_bl = isinstance(bl_pin, int)

        # When several panels share one RST line, only the FIRST should pulse it:
        # a later pulse would reset the already-initialised panel(s) too. The
        # backend passes do_hw_reset=False for subsequent panels on a shared RST.
        self._init_panel(rotation, invert, bgr, do_hw_reset=do_hw_reset)
        if self._bl is not None:
            self._bl.on()

    # ----- byte plumbing -----

    def _cmd(self, c: int) -> None:
        self._dc.off()
        self._spi.writebytes([c & 0xFF])

    def _data(self, data) -> None:
        self._dc.on()
        if isinstance(data, int):
            self._spi.writebytes([data & 0xFF])
        else:
            # writebytes2 chunks large buffers internally (handles >bufsiz).
            self._spi.writebytes2(data)

    # ----- init -----

    def _init_panel(self, rotation, invert, bgr, do_hw_reset=True):
        if self._rst is not None and do_hw_reset:
            self._rst.on(); time.sleep(0.05)
            self._rst.off(); time.sleep(0.05)
            self._rst.on(); time.sleep(0.15)

        # Per-panel software reset is always safe: it is addressed over this
        # panel's own chip-select, so it never disturbs the other panel.
        self._cmd(_SWRESET); time.sleep(0.15)
        self._cmd(_SLPOUT); time.sleep(0.12)

        madctl = _ROT.get(rotation, 0x00)
        if bgr:
            madctl |= _MADCTL_BGR
        self._cmd(_MADCTL); self._data(madctl)

        self._cmd(_COLMOD); self._data(0x55)        # 16 bit / pixel (RGB565)
        self._cmd(_INVON if invert else _INVOFF)    # IPS panels usually need INVON
        self._cmd(_NORON); time.sleep(0.01)
        self._cmd(_DISPON); time.sleep(0.05)

    # ----- addressing + blit -----

    def _set_window(self, x0, y0, x1, y1):
        xs = x0 + self.col_offset
        xe = x1 + self.col_offset
        ys = y0 + self.row_offset
        ye = y1 + self.row_offset
        self._cmd(_CASET)
        self._data(bytes([xs >> 8, xs & 0xFF, xe >> 8, xe & 0xFF]))
        self._cmd(_RASET)
        self._data(bytes([ys >> 8, ys & 0xFF, ye >> 8, ye & 0xFF]))
        self._cmd(_RAMWR)

    def blit(self, fb: np.ndarray, box=None) -> None:
        """Push framebuffer (or a sub-rectangle box=(x0,y0,x1,y1) inclusive) to RAM."""
        if box is None:
            x0, y0, x1, y1 = 0, 0, self.width - 1, self.height - 1
            sub = fb
        else:
            x0, y0, x1, y1 = box
            sub = fb[y0:y1 + 1, x0:x1 + 1]
        self._set_window(x0, y0, x1, y1)
        self._data(rgb_to_565_be(sub))

    def fill(self, rgb=(0, 0, 0)) -> None:
        fb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        fb[:] = np.array(rgb, dtype=np.uint8)
        self.blit(fb)

    def close(self) -> None:
        try:
            if self._bl is not None:
                self._bl.off()
                if self._owns_bl:
                    self._bl.close()
        finally:
            self._spi.close()
        # Shared DC/RST devices are owned and closed by the backend, not here.
