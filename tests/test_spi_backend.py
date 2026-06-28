"""Regression test for the dual-panel GPIO fix (bug #2), using fake hardware.

Without the fix, both panels allocate the shared DC(25)/RST(27) pins, and the
second allocation would raise gpiozero's GPIOPinInUse. We fake gpiozero + spidev
so this runs on any machine and assert: (a) construction succeeds, (b) exactly one
device per unique pin, (c) the shared RST line is pulsed only once.
"""
import sys
import types

import numpy as np


class FakeDOD:
    """Stand-in for gpiozero.DigitalOutputDevice that mimics pin-in-use errors."""
    instances: list["FakeDOD"] = []

    def __init__(self, pin):
        for d in FakeDOD.instances:
            if d.pin == pin and not d.closed:
                raise RuntimeError(f"GPIO busy: pin {pin} already in use")
        self.pin = pin
        self.closed = False
        self.on_calls = 0
        self.off_calls = 0
        FakeDOD.instances.append(self)

    def on(self):
        self.on_calls += 1

    def off(self):
        self.off_calls += 1

    def close(self):
        self.closed = True


class FakeSpiDev:
    def __init__(self):
        self.max_speed_hz = 0
        self.mode = 0

    def open(self, bus, dev):
        self.bus, self.dev = bus, dev

    def writebytes(self, data):
        pass

    def writebytes2(self, data):
        pass

    def close(self):
        pass


def test_spi_backend_shares_pins_without_conflict(monkeypatch):
    gz = types.ModuleType("gpiozero")
    gz.DigitalOutputDevice = FakeDOD
    sd = types.ModuleType("spidev")
    sd.SpiDev = FakeSpiDev
    monkeypatch.setitem(sys.modules, "gpiozero", gz)
    monkeypatch.setitem(sys.modules, "spidev", sd)
    FakeDOD.instances.clear()

    import robot_eyes.st7789 as st7789
    monkeypatch.setattr(st7789.time, "sleep", lambda *a, **k: None)  # no real delays
    from robot_eyes.backends import SpiBackend
    from robot_eyes.config import Config

    cfg = Config()
    backend = SpiBackend(cfg)              # must NOT raise GPIOPinInUse

    # exactly one device per unique pin: DC=25, RST=27 (shared) + BL 18, 24.
    pins = sorted(d.pin for d in FakeDOD.instances)
    assert pins == [18, 24, 25, 27]

    # shared RST pulsed exactly once (on/off/on => 2 on-calls), not per-panel.
    rst = next(d for d in FakeDOD.instances if d.pin == 27)
    assert rst.on_calls == 2

    # a frame can be pushed, and close() releases every device.
    fb = np.zeros((cfg.screen.height, cfg.screen.width, 3), dtype=np.uint8)
    backend.show(fb, fb)
    backend.close()
    assert all(d.closed for d in FakeDOD.instances)
