"""
memtools.py - Cross-platform process RAM measurement.

Answers "how much of my RAM does this use?" on Windows (your PC) and Linux (the
Pi 5) alike. Prefers psutil (accurate RSS everywhere); falls back to the stdlib
``resource`` module on Unix if psutil is missing.

  rss_mb()           -> current resident set size in MB
  MemorySampler      -> background thread that records the PEAK RSS while code runs
"""
from __future__ import annotations

import os
import threading
import time

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover - psutil normally present
    psutil = None
    _PROC = None


def rss_mb() -> float:
    """Current resident memory of this process, in MB (0.0 if unavailable)."""
    if _PROC is not None:
        return _PROC.memory_info().rss / (1024.0 * 1024.0)
    try:  # Unix fallback
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS bytes.
        import sys
        return ru / 1024.0 if sys.platform != "darwin" else ru / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def available_mb() -> float:
    """System RAM currently available, in MB (0.0 if unavailable)."""
    if psutil is not None:
        return psutil.virtual_memory().available / (1024.0 * 1024.0)
    return 0.0


def total_mb() -> float:
    """Total system RAM, in MB (0.0 if unavailable)."""
    if psutil is not None:
        return psutil.virtual_memory().total / (1024.0 * 1024.0)
    return 0.0


class MemorySampler:
    """Background sampler that tracks the peak RSS over a region of code.

        with MemorySampler() as s:
            ... run the pipeline ...
        print(s.peak_mb)
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.peak_mb = rss_mb()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            self._stop.wait(self.interval_s)

    def sample(self) -> float:
        """Take a manual sample now and return current RSS."""
        cur = rss_mb()
        self.peak_mb = max(self.peak_mb, cur)
        return cur

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.sample()
