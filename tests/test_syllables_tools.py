"""wvx.syllable_segmenters on synthetic audio: de Jong & Wempe nuclei (Parselmouth) and Sylber (PyTorch).

Skipped where those packages (the syllable venv, requirements-syllables.txt) or the pinned Sylber files
(scripts/syllable_segment.py fetch) are absent, e.g. on the Mac. No data.
"""

import numpy as np
import pytest

from wvx import syllables as sy

SR = 16000
CENTRES = [0.25, 0.55, 0.85, 1.15, 1.45]          # five voiced "syllables", 180 ms each, 120 ms quiet gaps


@pytest.fixture(scope="module")
def cfg():
    return sy.load_benchmark_config()


@pytest.fixture(scope="module")
def bursts(tmp_path_factory):
    import soundfile as sf

    t = np.arange(int(2.0 * SR)) / SR
    x = 1e-3 * np.random.default_rng(0).normal(size=t.size)
    for c in CENTRES:
        m = np.abs(t - c) < 0.09
        env = np.zeros_like(t)
        env[m] = np.hanning(m.sum())
        x += 0.3 * env * sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 8))
    p = tmp_path_factory.mktemp("audio") / "bursts.wav"
    sf.write(str(p), x.astype(np.float32), SR, subtype="PCM_16")
    return p


def test_praat_nuclei_published_algorithm(cfg, bursts):
    pytest.importorskip("parselmouth")
    from wvx.syllable_segmenters import praat_nuclei

    r = praat_nuclei(bursts, cfg["syllables"]["segmenters"]["nuclei"])
    # every burst is a voiced intensity peak with deep dips around it; the script never tests its last peak
    assert r["n_above_threshold"] == 5 and r["n_voiced"] == 4
    np.testing.assert_allclose(r["nuclei_s"], CENTRES[:4], atol=0.02)
    np.testing.assert_allclose(r["cuts_s"], [0.40, 0.70, 1.00], atol=0.03)       # minima in the gaps
    assert r["timecorrection"] == pytest.approx(1.0)


def test_praat_nuclei_ignores_unvoiced_peaks(cfg, tmp_path):
    pytest.importorskip("parselmouth")
    import soundfile as sf

    from wvx.syllable_segmenters import praat_nuclei

    t = np.arange(int(2.0 * SR)) / SR
    rng = np.random.default_rng(1)
    x = 1e-3 * rng.normal(size=t.size)
    for c in CENTRES:                                  # noise bursts: intensity peaks without pitch
        m = np.abs(t - c) < 0.09
        x[m] += 0.3 * np.hanning(m.sum()) * rng.normal(size=m.sum())
    sf.write(str(tmp_path / "noise.wav"), x.astype(np.float32), SR, subtype="PCM_16")
    r = praat_nuclei(tmp_path / "noise.wav", cfg["syllables"]["segmenters"]["nuclei"])
    assert r["n_dip_valid"] >= 3 and r["n_voiced"] == 0 and r["nuclei_s"] == [] and r["cuts_s"] == []


def test_sylber_segments_bursts(cfg, bursts):
    pytest.importorskip("torch")
    pytest.importorskip("sylber")
    from wvx.syllable_segmenters import SylberSegmenter, sylber_files

    scfg = cfg["syllables"]["segmenters"]["sylber"]
    ckpt, upstream = sylber_files(scfg, cfg["paths"]["sylber_models"])
    if not (ckpt.is_file() and (upstream / "config.json").is_file()):
        pytest.skip("run scripts/syllable_segment.py fetch")
    seg = SylberSegmenter(scfg, cfg["paths"]["sylber_models"], n_threads=2)
    assert seg.load_report["missing_keys"] == [] and seg.differs_from_package_defaults == {}
    out = np.asarray(seg(bursts)["segments_s"])
    assert out.shape == (5, 2)
    centres = out.mean(axis=1)
    np.testing.assert_allclose(centres, CENTRES, atol=0.05)
