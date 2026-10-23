"""Same-day inner -> attempted conversion (D-067/D-069): fit the transforms of ``wvx.neural_align`` on paired Kunz
blocks (attempted and imagined blocks holding the same cues), exactly as scripts/inner_projector.py does.

Trials' ``neural`` are the decoder inputs in whatever channel order the caller uses; every transform here is
equivariant to a channel permutation (per-channel gain; PCA, cosine DTW and whitening/rotation on all channels), so
fitting in the stored order and reordering afterwards equals reordering first (checked by
scripts/inner_transform_data.py).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from wvx import metrics, neural_align as na, neural_match as nm, targets as tg

TRANSFORMS = ("gain", "projector", "paired", "paired_shuffled")


def fit(pairs: Sequence[tuple[list, list]], settings: Mapping[str, Any], seed_key: int,
        names: Sequence[str] = TRANSFORMS) -> tuple[dict, dict]:
    """``pairs``: [(attempted trials, imagined trials)] of the fitting block pairs (same cues within a pair).
    settings: n_components, pool_bins, shrink, seed, dtw. Returns ({name: LinearMap}, {name: info})."""
    k, pool, seed = int(settings["n_components"]), int(settings["pool_bins"]), int(settings["seed"])
    dtw_cfg = dict(settings["dtw"])

    def frames(trials):
        return np.concatenate([nm.pool_frames(np.asarray(t.neural), pool) for t in trials])

    im_fr = frames([t for _, ims in pairs for t in ims])
    att_fr = frames([t for atts, _ in pairs for t in atts])
    V = na.pca_basis(np.concatenate([im_fr, att_fr]), k)
    maps, info = {}, {}
    if "gain" in names:
        maps["gain"] = na.fit_standardize(im_fr, att_fr)
        info["gain"] = maps["gain"].info()
    if "projector" in names:
        maps["projector"], info["projector"] = na.fit_projector(im_fr, att_fr, V)
    mu_i, mu_a = im_fr.mean(0), att_fr.mean(0)
    rng = np.random.default_rng([seed, seed_key])
    for name in ("paired", "paired_shuffled"):
        if name not in names:
            continue
        pi, pa = [], []
        for atts, ims in pairs:
            by_text = {tg.text_key(t.text): t for t in atts}
            partners = [by_text[tg.text_key(t.text)] for t in ims]
            if name == "paired_shuffled":                              # seeded derangement within the pair
                while True:
                    perm = rng.permutation(len(partners))
                    if not np.any(perm == np.arange(len(partners))):
                        break
                partners = [partners[j] for j in perm]
            for ti, ta in zip(ims, partners):
                fi, fa = nm.pool_frames(np.asarray(ti.neural), pool), nm.pool_frames(np.asarray(ta.neural), pool)
                ia, ib = metrics._align(((fi - mu_i) @ V.T).T, ((fa - mu_a) @ V.T).T, True,
                                        {"alignment": {"dtw": dtw_cfg}}, "none")
                pi.append(fi[ia]), pa.append(fa[ib])
        maps[name], info[name] = na.fit_paired_rotation(np.concatenate(pi), np.concatenate(pa), V,
                                                        float(settings["shrink"]), mu_i, mu_a)
    return maps, info
