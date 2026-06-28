import numpy as np

from affect_pi.memtools import MemorySampler, rss_mb, total_mb


def test_rss_is_positive():
    assert rss_mb() > 0.0


def test_total_ram_reported():
    # psutil is installed in this env; total should be a sane positive number.
    assert total_mb() > 0.0


def test_sampler_tracks_peak_growth():
    with MemorySampler(interval_s=0.005) as s:
        # allocate ~80 MB and touch it so it is resident.
        blocks = [np.ones((10_000_000,), dtype=np.float64) for _ in range(1)]
        blocks[0][::1000] = 2.0
        s.sample()
    assert s.peak_mb >= rss_mb() - 50  # peak is at least near current
    assert s.peak_mb > 0.0


def test_manual_sample_returns_current():
    s = MemorySampler()
    cur = s.sample()
    assert cur > 0.0
    assert s.peak_mb >= cur - 1e-6
