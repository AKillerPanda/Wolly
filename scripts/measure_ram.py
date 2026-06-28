#!/usr/bin/env python3
"""
measure_ram.py - How much RAM does the whole affect-pi stack use?

Runs the complete pipeline headlessly (no camera, synthetic frames) and prints a
stage-by-stage RSS breakdown plus the peak, so you know the real footprint on
your PC now and can predict it on the Pi 5 (4 GB).

    python scripts/measure_ram.py
    python scripts/measure_ram.py --frames 120     # longer run -> steadier peak

What it loads, in order (each line shows total RSS and the increase that stage
added): Python baseline -> OpenCV/numpy -> MediaPipe FaceLandmarker (.task) ->
emotion model (joblib) -> face registry + eye renderer/controller + emote policy
-> then runs N frames through landmarker + emotion + identity + eyes while a
background sampler records the peak.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from affect_pi.memtools import MemorySampler, available_mb, rss_mb, total_mb  # noqa: E402

_stages: list[tuple[str, float]] = []


def mark(name: str) -> None:
    _stages.append((name, rss_mb()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the RAM footprint of the full affect-pi pipeline.")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--landmarker-model", type=Path, default=Path("models/face_landmarker.task"))
    ap.add_argument("--emotion-model", type=Path, default=Path("artifacts/emotion_tasks_model.joblib"))
    args = ap.parse_args()

    mark("python baseline")

    import cv2  # noqa: F401
    import numpy as np
    mark("opencv + numpy")

    from affect_pi.mdistortion_tasks import detect_video, feature_from_pts, load_landmarker
    from affect_pi.face_identity import FaceRegistry, IdentityTracker
    from robot_eyes.behavior import EMOTES, AdaptivePolicy
    from robot_eyes.config import Config
    from robot_eyes.controller import EyeController
    from robot_eyes.renderer import EyeRenderer
    mark("import pipeline modules")

    if not args.landmarker_model.exists():
        print(f"NOTE: {args.landmarker_model} missing; skipping landmarker (RAM will be lower than live).")
        landmarker = None
    else:
        landmarker = load_landmarker(args.landmarker_model, video=True)
    mark("FaceLandmarker loaded")

    reader = None
    if args.emotion_model.exists():
        # Reuse the live app's reader so we measure the same objects.
        import importlib.util
        spec = importlib.util.spec_from_file_location("_glue", str(ROOT / "scripts" / "live_emotion_eyes.py"))
        glue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(glue)
        reader = glue.EmotionReader(args.emotion_model)
    else:
        print(f"NOTE: {args.emotion_model} missing; skipping emotion model.")
    mark("emotion model loaded")

    cfg = Config()
    # Load the real registry so identity matching is exercised, but disable
    # enrollment so this profiler never writes synthetic frames into known_faces.txt.
    registry = FaceRegistry.load(ROOT / "artifacts" / "known_faces.txt")
    identity = IdentityTracker(registry, auto_enroll=False)
    renderer = EyeRenderer(cfg)
    controller = EyeController(cfg, seed=0)
    policy = AdaptivePolicy(EMOTES, epsilon=0.2, seed=0, path=None)
    mark("eyes + identity + policy")

    rng = np.random.default_rng(0)
    with MemorySampler() as sampler:
        for i in range(args.frames):
            frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
            pts = detect_video(landmarker, frame, i * 33 + 1) if landmarker is not None else None
            if pts is None:  # synthetic noise rarely has a face; exercise downstream anyway
                pts = rng.uniform([120, 90], [520, 400], size=(478, 2)).astype(np.float32)
            if reader is not None:
                reader.predict(pts)
                reader.happiness()
            identity.update(pts)
            emote, _ = policy.select(identity.last_label or "anon", "neutral")
            policy.update(identity.last_label or "anon", "neutral", emote, 0.1)
            state = controller.update(1 / 30)
            renderer.render(replace(state, blink=controller.blink_left), "left")
            renderer.render(replace(state, blink=controller.blink_right), "right")
            sampler.sample()
    mark("after running frames")
    peak = sampler.peak_mb

    # ---- report ----
    print("\n================  affect-pi RAM footprint  ================")
    base = _stages[0][1]
    prev = base
    name_w = max(len(n) for n, _ in _stages)
    for name, rss in _stages:
        print(f"  {name:<{name_w}}  {rss:8.1f} MB   (+{rss - prev:6.1f})")
        prev = rss
    print("  " + "-" * (name_w + 28))
    print(f"  {'PEAK during run':<{name_w}}  {peak:8.1f} MB")
    print(f"  {'pipeline overhead':<{name_w}}  {peak - base:8.1f} MB   (peak minus python baseline)")
    if total_mb() > 0:
        print(f"\n  System RAM: {total_mb()/1024:.1f} GB total, {available_mb()/1024:.1f} GB free now.")
        print(f"  This run used about {100.0 * peak / total_mb():.1f}% of total RAM.")
        headroom = 4096.0  # Pi 5 4GB reference
        print(f"  On a 4 GB Pi 5 that peak would be ~{100.0 * peak / headroom:.0f}% of RAM "
              f"(before OS overhead).")
    print("==========================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
