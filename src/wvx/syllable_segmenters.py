"""Decoder-side syllable segmenters that need Praat or PyTorch (the syllable venv, requirements-syllables.txt).

Both read the LPCNet audio of a decoder output (``scripts/syllable_decode.py``) and return times in
seconds of audio; ``wvx.syllables`` turns them into frames on the neural timeline. Settings:
``syllables.segmenters`` in configs/experiments/syllable_benchmark.yaml, which holds the published
defaults; nothing here is tuned.

**Syllable nuclei** (``praat_nuclei``): de Jong, N. H. & Wempe, T. (2009). Praat script to detect
syllable nuclei and measure speech rate automatically. Behavior Research Methods 41(2), 385-390,
doi:10.3758/BRM.41.2.385. Reimplemented with Parselmouth from the algorithm of their script ("Praat
Script Syllable Nuclei", de Jong & Wempe 2008, updated 2010), step by step with the same Praat
commands and the form defaults for unfiltered sound:

1. subtract the mean; intensity contour (minimum pitch 50 Hz, automatic time step, mean subtracted);
2. threshold = median intensity + ignorance level (0 dB for unfiltered sound), at least the minimum;
3. candidate peaks = intensity maxima (Sinc70) whose value (cubic) exceeds the threshold;
4. a peak is kept if the intensity minimum between it and the NEXT peak lies more than the minimum dip
   (2 dB for unfiltered sound) below it. As in the script, the last peak is never tested, so it is
   never kept, and the reference intensity of every later peak is read from the intensity contour;
5. a kept peak is a nucleus if the pitch (autocorrelation; 0.02 s, 30-450 Hz, the script's settings)
   is defined at its time;
6. times are scaled by the script's ``timecorrection`` (sound duration / intensity-sound duration).

Our addition, needed to turn points into segments: between consecutive nuclei, the time of the
intensity minimum (``cuts_s``), found with the same contour and interpolation as step 4's dip.

**Sylber** (``SylberSegmenter``): Cho, C. J., Lee, N., Gupta, A., Agarwal, D., Chen, E., Black, A. W. &
Anumanchipalli, G. K. Sylber: Syllabic Embedding Representation of Speech from Raw Audio. ICLR 2025,
arXiv:2410.07168. The released package's ``Segmenter`` with its default thresholds and the released
checkpoint, loaded from a local copy pinned by revision and SHA-256 (``fetch_sylber``); one file per
call, through the package's own file loader (no padding, its own normalization).
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
from pathlib import Path
from typing import Any, Mapping, Union

PathLike = Union[str, Path]


def sha256(path: PathLike) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# --------------------------------------------------------------------------- syllable nuclei


def praat_nuclei(wav_path: PathLike, ncfg: Mapping[str, Any]) -> dict:
    """De Jong & Wempe syllable nuclei of one sound file (module docstring); times in seconds."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(wav_path))
    if snd.n_channels != 1:
        raise ValueError(f"{wav_path}: {snd.n_channels} channels, want mono")
    original_dur = call(snd, "Get total duration")
    call(snd, "Subtract mean")
    ic = ncfg["intensity"]
    intensity = call(snd, "To Intensity", float(ic["min_pitch_hz"]), float(ic["time_step_s"]), bool(ic["subtract_mean"]))
    min_int = call(intensity, "Get minimum", 0, 0, "Parabolic")
    med_int = call(intensity, "Get quantile", 0, 0, 0.5)
    threshold = med_int + float(ncfg["ignorance_level_db"])
    if threshold < min_int:
        threshold = min_int

    matrix = call(intensity, "Down to Matrix")
    snd_int = call(matrix, "To Sound (slice)", 1)
    int_dur = call(snd_int, "Get finishing time")
    ex = ncfg["peak_extrema"]
    pp = call(snd_int, "To PointProcess (extrema)", ex["channel"], bool(ex["include_maxima"]),
              bool(ex["include_minima"]), ex["interpolation"])
    n_extrema = call(pp, "Get number of points")
    times = [call(pp, "Get time from index", i + 1) for i in range(n_extrema)]
    vi = ncfg["peak_value_interpolation"]
    peaks = []
    for t in times:
        v = call(snd_int, "Get value at time", 1, t, vi)
        if v > threshold:
            peaks.append((t, v))

    valid = []
    if peaks:
        prec_t, prec_v = peaks[0]
        for p in range(len(peaks) - 1):
            foll_t = peaks[p + 1][0]
            dip = call(intensity, "Get minimum", prec_t, foll_t, ncfg["dip_interpolation"])
            if abs(prec_v - dip) > float(ncfg["min_dip_db"]):
                valid.append(peaks[p][0])
            prec_t = foll_t
            prec_v = call(intensity, "Get value at time", foll_t, vi)

    pc = ncfg["pitch_ac"]
    pitch = call(snd, "To Pitch (ac)", float(pc["time_step_s"]), float(pc["floor_hz"]), int(pc["max_candidates"]),
                 bool(pc["very_accurate"]), float(pc["silence_threshold"]), float(pc["voicing_threshold"]),
                 float(pc["octave_cost"]), float(pc["octave_jump_cost"]), float(pc["voiced_unvoiced_cost"]),
                 float(pc["ceiling_hz"]))
    pq = ncfg["pitch_query"]
    voiced = [t for t in valid if not math.isnan(call(pitch, "Get value at time", t, pq["unit"], pq["interpolation"]))]

    correction = original_dur / int_dur
    cuts = [call(intensity, "Get time of minimum", a, b, ncfg["cut_interpolation"]) for a, b in zip(voiced[:-1], voiced[1:])]
    if not all(math.isfinite(t) for t in voiced + cuts):
        raise ValueError(f"{wav_path}: undefined nucleus or cut time")
    return {
        "nuclei_s": [t * correction for t in voiced],
        "cuts_s": [t * correction for t in cuts],
        "duration_s": original_dur,
        "threshold_db": threshold if math.isfinite(threshold) else None,   # undefined for digital silence
        "timecorrection": correction,
        "n_extrema": int(n_extrema),
        "n_above_threshold": len(peaks),
        "n_dip_valid": len(valid),
        "n_voiced": len(voiced),
    }


def parselmouth_versions() -> dict:
    import parselmouth

    return {"praat-parselmouth": parselmouth.__version__, "praat": parselmouth.PRAAT_VERSION}


# ------------------------------------------------------------------------------------- Sylber


def sylber_files(scfg: Mapping[str, Any], models_root: PathLike) -> tuple[Path, Path]:
    """(checkpoint file, directory with the HuBERT config.json) of the local, pinned copy."""
    root = Path(models_root)
    ckpt = root / scfg["hf_repo"].replace("/", "_") / scfg["checkpoint"]
    upstream = root / scfg["upstream_repo"].replace("/", "_")
    return ckpt, upstream


def fetch_sylber(scfg: Mapping[str, Any], models_root: PathLike) -> dict:
    """Download the pinned checkpoint and HuBERT config (if absent) and verify both SHA-256 sums."""
    from huggingface_hub import hf_hub_download

    ckpt, upstream = sylber_files(scfg, models_root)
    out = {}
    for repo, rev, name, dest, want in (
        (scfg["hf_repo"], scfg["revision"], scfg["checkpoint"], ckpt.parent, scfg["checkpoint_sha256"]),
        (scfg["upstream_repo"], scfg["upstream_revision"], "config.json", upstream, scfg["upstream_config_sha256"]),
    ):
        path = dest / name
        if not path.is_file():
            hf_hub_download(repo, name, revision=rev, local_dir=str(dest))
        got = sha256(path)
        if got != want:
            raise RuntimeError(f"{path}: SHA-256 {got}, expected {want}")
        out[f"{repo}/{name}"] = {"revision": rev, "sha256": got, "bytes": path.stat().st_size}
    return out


class SylberSegmenter:
    """The released ``sylber.Segmenter`` on CPU, from the local pinned files, offline."""

    def __init__(self, scfg: Mapping[str, Any], models_root: PathLike, n_threads: int):
        os.environ["HF_HUB_OFFLINE"] = "1"            # never reach the network during a run
        import torch

        torch.set_num_threads(int(n_threads))
        import sylber
        from sylber import Segmenter

        ckpt, upstream = sylber_files(scfg, models_root)
        for path, key in ((ckpt, "checkpoint_sha256"), (upstream / "config.json", "upstream_config_sha256")):
            if not path.is_file():
                raise FileNotFoundError(f"{path} missing: run scripts/syllable_segment.py fetch")
            if sha256(path) != scfg[key]:
                raise RuntimeError(f"{path}: SHA-256 does not match {key}")
        settings = {k: scfg[k] for k in ("encoding_layer", "merge_threshold", "norm_threshold")}
        defaults = {k: v.default for k, v in inspect.signature(Segmenter.__init__).parameters.items() if k in settings}
        self.differs_from_package_defaults = {k: (settings[k], defaults.get(k)) for k in settings
                                              if settings[k] != defaults.get(k)}
        self.segmenter = Segmenter(model_ckpt=str(ckpt), speech_upstream=str(upstream), device="cpu", **settings)
        # Segmenter loads with strict=False: reload to see that every HuBERT weight came from the checkpoint.
        state = torch.load(str(ckpt), map_location="cpu")
        res = self.segmenter.speech_model.load_state_dict(state, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"Sylber checkpoint lacks {len(res.missing_keys)} model weights, e.g. {res.missing_keys[:3]}")
        self.load_report = {"missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys),
                            "n_checkpoint_tensors": len(state)}
        self.versions = {"sylber": getattr(sylber, "__version__", None) or _dist_version("sylber"),
                         "torch": torch.__version__, "torchaudio": _dist_version("torchaudio"),
                         "transformers": _dist_version("transformers")}

    def __call__(self, wav_path: PathLike) -> dict:
        out = self.segmenter(wav_file=str(wav_path), in_second=True)
        seg = out["segments"]
        return {"segments_s": [[float(a), float(b)] for a, b in (seg.tolist() if len(seg) else [])]}


def _dist_version(name: str):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
