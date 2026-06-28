"""
face_identity.py - Recognise and enroll faces by their innate facial-shape matrix.

The "identity" of a face here is an **expression-invariant** signature: the upper
triangle of the normalized pairwise-distance matrix among a set of *rigid*
landmarks (forehead, nose bridge/tip, eye corners, cheekbones, temples). Those
points barely move when you smile/frown, unlike the mouth/eyelid points the
emotion Mdistortion uses -- so this captures bone structure / proportions, i.e.
who the face is, not what it is expressing.

Why pairwise distances: they are translation- and rotation-invariant by
construction, and dividing by the cheek-to-cheek width makes them scale-invariant
too. The remaining weakness is out-of-plane head rotation (yaw/pitch), so enroll
roughly frontal. This is a lightweight geometric recogniser -- good enough to tell
a handful of people apart in similar pose/lighting, not a secure biometric.

Storage is a plain, human-readable text file (one record per known face):

    # id | label | n_samples | comma-separated signature floats
    1 | Anuska | 14 | 0.812,0.447,...

so the stored "matrices of facial shapes" can be inspected and edited by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Cheek-to-cheek width is the scale denominator (same pair the emotion code uses).
LEFT_CHEEK_ID = 234
RIGHT_CHEEK_ID = 454

# Rigid, expression-stable landmarks spread across the face. Deliberately excludes
# mouth, eyelids and jaw/chin (which move with expression or mouth opening).
STABLE_IDS: list[int] = [
    10, 9, 8,            # forehead centre / glabella
    168, 6, 197, 195, 4, 1,   # nose bridge down to nose tip
    33, 133,             # left eye outer / inner corner
    362, 263,            # right eye inner / outer corner
    234, 454,            # left / right cheekbone
    127, 356,            # left / right temple
]

_TRI = np.triu_indices(len(STABLE_IDS), k=1)
SIGNATURE_DIM = len(_TRI[0])   # 17 points -> 136 pairwise distances


def identity_signature(pts: np.ndarray | None) -> np.ndarray | None:
    """Normalized pairwise-distance signature for a face, or None if unusable.

    `pts` is the (N>=478, 2) array of landmark pixel coordinates from the
    FaceLandmarker. Returns a (SIGNATURE_DIM,) float32 vector.
    """
    if pts is None or len(pts) <= max(STABLE_IDS):
        return None
    scale = float(np.linalg.norm(pts[LEFT_CHEEK_ID] - pts[RIGHT_CHEEK_ID]))
    if scale < 1e-6:
        return None
    p = pts[STABLE_IDS].astype(np.float64)
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) / scale
    return dist[_TRI].astype(np.float32)


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference between two signatures. Same person -> small, different -> large."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


@dataclass
class FaceRecord:
    id: int
    label: str
    n_samples: int
    signature: np.ndarray   # running mean signature (the stored "face shape matrix")


@dataclass
class FaceRegistry:
    """In-memory set of known faces backed by a human-readable text file."""

    path: Path
    records: list[FaceRecord] = field(default_factory=list)

    # ----- persistence -----

    @classmethod
    def load(cls, path: str | Path) -> "FaceRegistry":
        path = Path(path)
        reg = cls(path=path)
        if not path.exists():
            return reg
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4:
                continue
            rid, label, n, vec = parts
            sig = np.array([float(v) for v in vec.split(",") if v], dtype=np.float32)
            reg.records.append(FaceRecord(int(rid), label, int(n), sig))
        return reg

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# affect-pi known faces. Each record is an expression-invariant face-shape",
            "# signature: upper triangle of the normalized pairwise-distance matrix among",
            f"# {len(STABLE_IDS)} rigid landmarks ({SIGNATURE_DIM} values).",
            "# format: id | label | n_samples | comma-separated signature floats",
        ]
        for r in self.records:
            vec = ",".join(f"{v:.5f}" for v in r.signature)
            lines.append(f"{r.id} | {r.label} | {r.n_samples} | {vec}")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ----- matching / enrollment -----

    def match(self, signature: np.ndarray, threshold: float) -> tuple[FaceRecord | None, float]:
        """Return (best_record, distance) if within threshold, else (None, distance)."""
        if not self.records:
            return None, float("inf")
        dists = [signature_distance(signature, r.signature) for r in self.records]
        i = int(np.argmin(dists))
        best = dists[i]
        return (self.records[i] if best <= threshold else None), best

    def add(self, signature: np.ndarray, label: str, n_samples: int = 1) -> FaceRecord:
        new_id = (max((r.id for r in self.records), default=0) + 1)
        rec = FaceRecord(new_id, label, n_samples, np.asarray(signature, dtype=np.float32))
        self.records.append(rec)
        return rec

    def reinforce(self, record: FaceRecord, signature: np.ndarray) -> None:
        """Refine a known face's stored signature with a new sighting (online mean)."""
        n = record.n_samples
        record.signature = ((record.signature * n + np.asarray(signature, np.float32)) / (n + 1)).astype(np.float32)
        record.n_samples = n + 1

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class IdentityTracker:
    """Per-frame recognition with temporal stability and auto-enrollment.

    Feed it landmark points each frame; it reports who it sees. An unknown face
    that persists for `enroll_after` frames is averaged and stored automatically,
    so the system "remembers a new face the first time it sees it for a moment".
    """

    registry: FaceRegistry
    match_threshold: float = 0.045
    enroll_after: int = 20
    auto_enroll: bool = True
    reinforce: bool = True

    _unknown_buf: list[np.ndarray] = field(default_factory=list)
    last_status: str = "starting"
    last_label: str | None = None
    last_distance: float = float("inf")

    def _auto_label(self) -> str:
        return f"user{len(self.registry) + 1}"

    def update(self, pts: np.ndarray | None) -> str:
        """Return a short status string, e.g. 'Anuska (0.021)' or 'enrolling 7/20'."""
        sig = identity_signature(pts)
        if sig is None:
            self.last_status = "no face"
            self.last_label = None
            return self.last_status

        rec, dist = self.registry.match(sig, self.match_threshold)
        self.last_distance = dist
        if rec is not None:
            self._unknown_buf.clear()
            if self.reinforce:
                self.registry.reinforce(rec, sig)
            self.last_label = rec.label
            self.last_status = f"{rec.label} ({dist:.3f})"
            return self.last_status

        # Unknown face.
        self.last_label = None
        if not self.auto_enroll:
            self.last_status = f"unknown ({dist:.3f})"
            return self.last_status

        self._unknown_buf.append(sig)
        if len(self._unknown_buf) >= self.enroll_after:
            mean_sig = np.mean(np.stack(self._unknown_buf, axis=0), axis=0)
            new = self.registry.add(mean_sig, self._auto_label(), n_samples=len(self._unknown_buf))
            self.registry.save()
            self._unknown_buf.clear()
            self.last_label = new.label
            self.last_status = f"enrolled {new.label}"
            return self.last_status

        self.last_status = f"enrolling {len(self._unknown_buf)}/{self.enroll_after}"
        return self.last_status
