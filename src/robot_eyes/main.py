#!/usr/bin/env python3
"""
main.py - Run the robot eyes on hardware or in the desktop simulator.

Examples:
    python -m robot_eyes.main                 # auto-detect (Pi 5 -> hardware, else sim)
    python -m robot_eyes.main --sim           # force simulator (your PC / VS Code)
    python -m robot_eyes.main --hardware      # force SPI panels
    python -m robot_eyes.main --calibrate     # draw a border+corner test pattern (hardware)
    python -m robot_eyes.main --sim --rss     # print peak RSS each second

Controls in the simulator window:
    1-5 = mood (neutral/happy/angry/tired/surprised)
    arrows = look, c = centre, space = blink, q/esc = quit
"""
from __future__ import annotations

import argparse
import time
import sys

from .config import Config, Mood
from .renderer import EyeRenderer, RenderState
from .controller import EyeController


def is_pi5() -> bool:
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return b"Raspberry Pi 5" in f.read()
    except OSError:
        return False


def peak_rss_mb() -> float:
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return ru / 1024.0 if sys.platform != "darwin" else ru / (1024.0 * 1024.0)


# Demo schedule: (mood, seconds). Runs on hardware (no keyboard) and as sim default.
DEMO = [
    (Mood.NEUTRAL, 3.0), (Mood.HAPPY, 3.0), (Mood.SURPRISED, 2.5),
    (Mood.ANGRY, 3.0), (Mood.TIRED, 3.0), (Mood.NEUTRAL, 2.0),
]


def calibrate(cfg: Config) -> None:
    """Hardware-only: 1px white border + R/G/B/W corner blocks to verify
    orientation, col/row offsets and colour order. If the border is clipped or
    wrapped, adjust col_offset/row_offset; if colours are wrong, toggle hw.bgr."""
    import numpy as np
    from .backends import SpiBackend
    H, W = cfg.screen.height, cfg.screen.width
    fb = np.zeros((H, W, 3), dtype=np.uint8)
    fb[0, :] = fb[-1, :] = fb[:, 0] = fb[:, -1] = (255, 255, 255)
    s = 16
    fb[1:s, 1:s] = (255, 0, 0)        # top-left  = RED
    fb[1:s, -s:-1] = (0, 255, 0)      # top-right = GREEN
    fb[-s:-1, 1:s] = (0, 0, 255)      # bot-left  = BLUE
    fb[-s:-1, -s:-1] = (255, 255, 255)
    be = SpiBackend(cfg, use_dirty_rect=False)
    try:
        be.show(fb, fb)
        print("Calibration pattern sent. Expect: full 1px white border, "
              "RED top-left, GREEN top-right, BLUE bottom-left. Ctrl-C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        be.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dual-display pixelated robot eyes.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="force desktop simulator")
    g.add_argument("--hardware", action="store_true", help="force SPI panels")
    ap.add_argument("--calibrate", action="store_true", help="hardware test pattern")
    ap.add_argument("--rss", action="store_true", help="print peak RSS once per second")
    ap.add_argument("--no-dirty", action="store_true", help="disable dirty-rect (full-frame) SPI writes")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for blinks/saccades")
    args = ap.parse_args(argv)

    cfg = Config()

    use_hw = args.hardware or (not args.sim and is_pi5())

    if args.calibrate:
        if not use_hw:
            print("--calibrate requires hardware (run on the Pi with panels attached).")
            return 2
        calibrate(cfg)
        return 0

    renderer = EyeRenderer(cfg)
    controller = EyeController(cfg, seed=args.seed)

    if use_hw:
        from .backends import SpiBackend
        backend = SpiBackend(cfg, use_dirty_rect=not args.no_dirty)
    else:
        from .backends import SimBackend
        backend = SimBackend(cfg)

    target_dt = 1.0 / cfg.anim.target_fps
    demo_i, demo_t = 0, 0.0
    controller.set_mood(DEMO[0][0])
    manual_mood = False   # once the user presses a mood key, stop the demo schedule

    last = time.perf_counter()
    last_rss = last
    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now

            # --- input (simulator only) ---
            quit_ = False
            for ev in backend.events():
                if ev == "quit":
                    quit_ = True
                elif ev == "blink":
                    controller.blink()
                elif ev.startswith("mood:"):
                    controller.set_mood(Mood(ev.split(":", 1)[1]))
                    manual_mood = True
                elif ev.startswith("look:"):
                    dx, dy = (float(v) for v in ev.split(":", 1)[1].split(","))
                    controller.look(dx, dy, hold=(dx != 0 or dy != 0))
            if quit_:
                break

            # --- demo mood schedule (until user takes manual control) ---
            if not manual_mood:
                demo_t += dt
                if demo_t >= DEMO[demo_i][1]:
                    demo_t = 0.0
                    demo_i = (demo_i + 1) % len(DEMO)
                    controller.set_mood(DEMO[demo_i][0])

            state: RenderState = controller.update(dt)
            left = renderer.render(state, cfg.hw.eye0.eye_side)
            right = renderer.render(state, cfg.hw.eye1.eye_side)
            backend.show(left, right)

            if args.rss and now - last_rss >= 1.0:
                last_rss = now
                print(f"peak RSS = {peak_rss_mb():.1f} MB")

            # --- frame cap ---
            sleep = target_dt - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        if args.rss:
            print(f"final peak RSS = {peak_rss_mb():.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
