#!/usr/bin/env python3
"""
retrain_on_pc.py - Refresh the bundled emotion model on your PC, then sync it
into the emotion_src/ deploy bundle so you can re-deploy to the Pi.

Training belongs on the PC (the Pi has neither the dataset nor the CPU budget),
so this is the "--retrain-on-PC" helper referenced by emotion_src/README_PI.md.
It runs the same trainer as `affect-pi-train-emotion`, then copies the fresh
model into emotion_src/artifacts/ and re-zips the bundle.

    # full refresh (recommended before deploying)
    python scripts/retrain_on_pc.py --max-per-class 400 --de-iterations 8

    # quick low-data refresh (rough, for testing the flow)
    python scripts/retrain_on_pc.py --max-per-class 60 --no-de

Then copy emotion_src/ (or emotion_src.zip) to the Pi again -- or just scp the one
file emotion_src/artifacts/emotion_tasks_model.joblib over the old one.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain the emotion model on the PC and refresh the deploy bundle.")
    p.add_argument("--emotions-dir", type=Path, default=REPO / "data" / "emotions" / "Data")
    p.add_argument("--landmarker-model", type=Path, default=REPO / "models" / "face_landmarker.task")
    p.add_argument("--max-per-class", type=int, default=400, help="Images per class (more = better, slower)")
    p.add_argument("--de-iterations", type=int, default=8, help="DE generations (0 / --no-de to skip)")
    p.add_argument("--de-popsize", type=int, default=16)
    p.add_argument("--no-de", action="store_true", help="Skip DE node-group optimization")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-model", type=Path, default=REPO / "artifacts" / "emotion_tasks_model.joblib")
    p.add_argument("--output-report", type=Path, default=REPO / "artifacts" / "emotion_tasks_report.json")
    p.add_argument("--output-ranges", type=Path, default=REPO / "artifacts" / "emotion_tasks_ranges.json")
    p.add_argument("--bundle-dir", type=Path, default=REPO / "emotion_src", help="Deploy bundle to refresh")
    p.add_argument("--zip-path", type=Path, default=REPO / "emotion_src.zip")
    p.add_argument("--no-zip", action="store_true", help="Update the folder but skip re-zipping")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.emotions_dir.exists():
        raise SystemExit(f"Training data not found: {args.emotions_dir}. Run this on the PC (with data/), not the Pi.")
    if not args.landmarker_model.exists():
        raise SystemExit(f"FaceLandmarker model not found: {args.landmarker_model}.")

    cmd = [
        sys.executable, "-m", "affect_pi.train_emotion_tasks",
        "--emotions-dir", str(args.emotions_dir),
        "--landmarker-model", str(args.landmarker_model),
        "--max-per-class", str(args.max_per_class),
        "--de-popsize", str(args.de_popsize),
        "--de-iterations", "0" if args.no_de else str(args.de_iterations),
        "--seed", str(args.seed),
        "--output-model", str(args.output_model),
        "--output-report", str(args.output_report),
        "--output-ranges", str(args.output_ranges),
    ]
    if args.no_de:
        cmd.append("--no-de")

    print(f">> Training: {' '.join(cmd[2:])}\n")
    # Inherit stdout so landmarking/DE progress is visible. cwd=REPO so `affect_pi`
    # resolves via the editable install regardless of where this is launched.
    result = subprocess.run(cmd, cwd=str(REPO))
    if result.returncode != 0:
        raise SystemExit(f"Training failed (exit {result.returncode}).")
    if not args.output_model.exists():
        raise SystemExit("Training reported success but no model file was written.")

    # --- sync into the deploy bundle ---
    bundle_artifacts = args.bundle_dir / "artifacts"
    if not bundle_artifacts.exists():
        raise SystemExit(f"Bundle not found at {args.bundle_dir} (expected an artifacts/ subfolder).")
    dest = bundle_artifacts / "emotion_tasks_model.joblib"
    shutil.copy2(args.output_model, dest)
    print(f"\n>> Refreshed bundle model: {dest}")

    if not args.no_zip:
        base = args.zip_path.with_suffix("")   # make_archive adds .zip
        shutil.make_archive(str(base), "zip", root_dir=str(args.bundle_dir.parent),
                            base_dir=args.bundle_dir.name)
        print(f">> Re-zipped bundle: {args.zip_path}")

    # --- summary ---
    try:
        report = json.loads(args.output_report.read_text(encoding="utf-8"))
        acc = report.get("training", {}).get("accuracy")
        classes = report.get("classes")
        used = report.get("n_samples_used")
        print(f"\n>> New model: accuracy={acc:.3f} on {used} samples, classes={classes}")
    except Exception:
        pass

    print("\n>> Next: re-deploy to the Pi. Either copy the whole emotion_src/ again,")
    print("   or just sync the one file:")
    print(f"     scp {dest}  pi@<pi-ip>:~/emotion_src/artifacts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
