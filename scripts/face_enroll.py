#!/usr/bin/env python3
"""
face_enroll.py - Enroll / list / recognise faces in the known-faces text file.

This is the standalone tester for integration step #2 (face memory). It uses the
same FaceLandmarker as the rest of the pipeline and the expression-invariant
identity signature in affect_pi.face_identity.

    # capture ~30 frames of your face and store it as "Anuska"
    python scripts/face_enroll.py enroll --label Anuska

    # print the known-faces file
    python scripts/face_enroll.py list

    # live window: shows who it recognises + distance (calibrate the threshold)
    python scripts/face_enroll.py recognize --mirror

Default store: artifacts/known_faces.txt  (override with --faces-file)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from affect_pi.mdistortion_tasks import DEFAULT_MODEL, detect_video, load_landmarker  # noqa: E402
from affect_pi.face_identity import (  # noqa: E402
    FaceRegistry,
    IdentityTracker,
    identity_signature,
    signature_distance,
)

DEFAULT_FACES = Path("artifacts/known_faces.txt")


def open_camera(index: int):
    for backend, name in [(cv2.CAP_DSHOW, "dshow"), (cv2.CAP_MSMF, "msmf"), (cv2.CAP_ANY, "default")]:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok = False
            for _ in range(8):
                ok, _ = cap.read()
                if ok:
                    break
            if ok:
                return cap, name
        cap.release()
    return None, None


def _landmarker(model_path: Path):
    if not model_path.exists():
        raise SystemExit(f"FaceLandmarker model not found: {model_path} (see README to download).")
    return load_landmarker(model_path, video=True)


def cmd_list(args) -> int:
    reg = FaceRegistry.load(args.faces_file)
    if not reg.records:
        print(f"No known faces in {args.faces_file}.")
        return 0
    print(f"{len(reg)} known face(s) in {args.faces_file}:")
    for r in reg.records:
        print(f"  id={r.id}  label={r.label!r}  samples={r.n_samples}  sig_dim={r.signature.size}")
    return 0


def cmd_enroll(args) -> int:
    reg = FaceRegistry.load(args.faces_file)
    landmarker = _landmarker(args.landmarker_model)
    cap, backend = open_camera(args.camera)
    if cap is None:
        raise SystemExit(f"Could not open camera {args.camera}.")
    print(f"Camera (backend={backend}). Look frontally; capturing {args.frames} good frames...")

    start = time.time()
    last_ts = -1
    sigs: list[np.ndarray] = []
    try:
        while len(sigs) < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            ts = max(int((time.time() - start) * 1000), last_ts + 1)
            last_ts = ts
            pts = detect_video(landmarker, frame, ts)
            sig = identity_signature(pts)
            if sig is not None:
                sigs.append(sig)
            n = len(sigs)
            cv2.putText(frame, f"capturing {n}/{args.frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if pts is not None else (0, 0, 255), 2)
            cv2.imshow("enroll (q to abort)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    if len(sigs) < max(5, args.frames // 3):
        raise SystemExit(f"Only captured {len(sigs)} usable frames; aborting (face not detected enough).")

    mean_sig = np.mean(np.stack(sigs, axis=0), axis=0)
    # Report self-consistency so you can sanity-check the threshold.
    spread = float(np.mean([signature_distance(s, mean_sig) for s in sigs]))
    rec, dist = reg.match(mean_sig, threshold=args.match_threshold)
    if rec is not None and not args.force:
        print(f"This face already matches '{rec.label}' (distance {dist:.3f} <= {args.match_threshold}). "
              "Use --force to add anyway.")
        return 1
    new = reg.add(mean_sig, args.label, n_samples=len(sigs))
    reg.save()
    print(f"Enrolled id={new.id} label={new.label!r} from {len(sigs)} frames "
          f"(intra-face spread {spread:.4f}). Saved -> {args.faces_file}")
    if reg.records[:-1]:
        nearest = min(((signature_distance(mean_sig, r.signature), r.label) for r in reg.records[:-1]))
        print(f"Nearest other face: '{nearest[1]}' at distance {nearest[0]:.4f} "
              f"(should be comfortably above the intra-face spread).")
    return 0


def cmd_recognize(args) -> int:
    reg = FaceRegistry.load(args.faces_file)
    tracker = IdentityTracker(reg, match_threshold=args.match_threshold,
                              enroll_after=args.enroll_after, auto_enroll=args.auto_enroll)
    landmarker = _landmarker(args.landmarker_model)
    cap, backend = open_camera(args.camera)
    if cap is None:
        raise SystemExit(f"Could not open camera {args.camera}.")
    print(f"Camera (backend={backend}). Known faces: {[r.label for r in reg.records]}. q to quit.")

    start = time.time()
    last_ts = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            ts = max(int((time.time() - start) * 1000), last_ts + 1)
            last_ts = ts
            pts = detect_video(landmarker, frame, ts)
            status = tracker.update(pts)
            color = (0, 200, 255) if tracker.last_label else (0, 0, 255)
            cv2.putText(frame, f"who: {status}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"threshold {args.match_threshold:.3f}  nearest {tracker.last_distance:.3f}",
                        (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("recognize (q quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Enroll / list / recognise faces in the known-faces text file.")
    ap.add_argument("--faces-file", type=Path, default=DEFAULT_FACES)
    ap.add_argument("--landmarker-model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--match-threshold", type=float, default=0.045)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    pe = sub.add_parser("enroll")
    pe.add_argument("--label", required=True, help="Name to store this face under")
    pe.add_argument("--frames", type=int, default=30, help="Good frames to average")
    pe.add_argument("--force", action="store_true", help="Enroll even if it matches a known face")

    pr = sub.add_parser("recognize")
    pr.add_argument("--auto-enroll", action="store_true", help="Auto-store unknown faces as userN")
    pr.add_argument("--enroll-after", type=int, default=20)

    args = ap.parse_args()
    return {"list": cmd_list, "enroll": cmd_enroll, "recognize": cmd_recognize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
