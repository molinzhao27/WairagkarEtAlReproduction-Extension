#!/usr/bin/env python
"""Kunz features after ``wvx.preprocess`` vs Wairagkar's released features. Runs on the GPU host, CPU only.

    python -I scripts/compare_features.py [--config configs/data/kunz.yaml]

Settings: ``adapter.feature_check`` (and ``adapter.normalization``) in configs/data/kunz.yaml.
Writes one small JSON (``feature_check.out``) with, separately for spike-band power (SBP) and threshold
crossings (TX), go-period trials only:

- ``comparison``: pooled value ranges and quantiles, per-channel mean / s.d. distributions, lag-k
  autocorrelation (smoothing signature) and per-array (64-channel) summaries, for Wairagkar day 482
  (all control-experiment blocks), each Kunz condition with the configured preprocessing, and each Kunz
  sensitivity variant. Wairagkar day 482 is a different day from Kunz 2024.03.03, so close but not
  identical numbers are expected.
- ``identification``: what the released features themselves say about their preprocessing (used to
  set the defaults in the config, independent of Kunz): the impulse response to an isolated threshold
  crossing (kernel shape and length), the spacing of count levels after deconvolving the configured
  kernel (log offset), and rolling-window hypotheses tested against counts recovered from those
  levels. The same impulse/offset checks on the processed Kunz features verify our implementation.
- ``channel_order``: which electrode order Kunz is in, judged against Wairagkar day 195 (the released
  session nearest in time): per-channel threshold-crossing rate (recovered from the processed features)
  and the channel-by-channel correlation matrices of SBP and TX, with the Kunz columns as they are,
  reordered the way their ``reorder_channels`` does (``out[:, j] = in[:, m[m[j]]]``, see
  ``wvx.inference.reorder_channels``), and reordered by the inverse of that. Chance level: random
  shuffles within each 64-channel array.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.stats import skew, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import load_config, load_yaml, resolve_path  # noqa: E402
from wvx.kunz import load_condition  # noqa: E402
from wvx.preprocess import block_runs, rolling_zscore, sigmoid_kernel  # noqa: E402
from wvx.seeding import set_seed  # noqa: E402

FEATS = ("spikepow", "threshcross")


def _r(x, nd=5):
    """Round floats (recursively) so the JSON stays small and readable."""
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    return x


# ----------------------------------------------------------------------------- data

def load_wairagkar(fc: dict) -> dict:
    files = sorted(resolve_path(fc["wairagkar_dir"]).glob(fc["wairagkar_glob"]))
    if not files:
        raise FileNotFoundError(f"no {fc['wairagkar_glob']} in {fc['wairagkar_dir']}")
    out = {"files": [f.name for f in files], "spikepow": [], "threshcross": []}
    for f in files:
        m = scipy.io.loadmat(f, variable_names=["spikepow_trials", "threshcross_trials"])
        for feat in FEATS:
            out[feat] += [np.asarray(t, np.float64) for t in m[f"{feat}_trials"].ravel()]
    return out


def load_kunz(cfg: dict, cond: str, adapter: dict, n_ch: int) -> dict:
    spec = adapter["conditions"][cond]
    s = load_condition(resolve_path(cfg["output_dir"]) / spec["file"], cond, spec["layer"], adapter)
    trials = [t.neural.astype(np.float64) for t in s.trials]
    return {"spikepow": [t[:, :n_ch] for t in trials], "threshcross": [t[:, n_ch:] for t in trials]}


def kunz_raw_sbp_zscore(cfg: dict, cond: str, adapter: dict) -> list:
    """Go-period trials of the rolling z-score of SBP WITHOUT the log and without smoothing. Compare with
    ``identification.*.sbp_deconvolved`` (their features with the smoothing undone) to check the log on SBP."""
    roll = adapter["normalization"]["params"]["rolling"]
    m = scipy.io.loadmat(resolve_path(cfg["output_dir"]) / adapter["conditions"][cond]["file"],
                         variable_names=["spikePow", "blockNum", "goTrialEpochs"])
    sp = np.minimum(m["spikePow"].astype(np.float64), adapter["clip_spikepow"])
    blocks = m["blockNum"].ravel()
    z = np.empty_like(sp)
    for run in block_runs(blocks):
        z[run] = rolling_zscore(sp[run], int(roll["window_ms"] // adapter["bin_ms"]), roll["include_current"],
                                roll["ddof"], roll["min_std"], roll["zero_std_value"])
    return [z[g0 - 1:g1] for g0, g1 in m["goTrialEpochs"].astype(int)]


# ----------------------------------------------------------------------------- statistics

def acf(trials: list, lags: list) -> np.ndarray:
    """Per-channel autocorrelation at each lag, pairs only within trials, around the pooled channel mean."""
    mean = np.concatenate(trials).mean(0)
    num = np.zeros((len(lags), trials[0].shape[1]))
    den = np.zeros(trials[0].shape[1])
    for t in trials:
        c = t - mean
        den += (c * c).sum(0)
        for i, lag in enumerate(lags):
            if t.shape[0] > lag:
                num[i] += (c[lag:] * c[:-lag]).sum(0)
    return num / den


def describe(trials: list, fc: dict, arrays: list) -> dict:
    q, lags, cpa = fc["quantiles"], fc["lags"], fc["channels_per_array"]
    x = np.concatenate(trials)
    ch_mean, ch_sd = x.mean(0), x.std(0)
    ac = acf(trials, lags)
    out = {
        "n_trials": len(trials), "n_bins": int(x.shape[0]),
        "pooled": {"mean": x.mean(), "sd": x.std(), "min": x.min(), "max": x.max(),
                   "quantiles": dict(zip(map(str, q), np.quantile(x, q)))},
        "channel_mean_quantiles": dict(zip(map(str, q), np.quantile(ch_mean, q))),
        "channel_sd_quantiles": dict(zip(map(str, q), np.quantile(ch_sd, q))),
        "channel_skew_median": float(np.median(skew(x, axis=0))),
        "acf_median": {str(lag): float(np.median(a)) for lag, a in zip(lags, ac)},
        "acf_iqr": {str(lag): np.quantile(a, [0.25, 0.75]) for lag, a in zip(lags, ac)},
        "per_array": [],
    }
    for i, name in enumerate(arrays):
        sl = slice(i * cpa, (i + 1) * cpa)
        out["per_array"].append({
            "array": name, "channel_mean_median": np.median(ch_mean[sl]),
            "channel_sd_median": np.median(ch_sd[sl]),
            "acf_median": {str(lag): float(np.median(a[sl])) for lag, a in zip(lags, ac)},
        })
    return out


# ----------------------------------------------------------------------------- identification

def impulse_response(trials: list, idc: dict) -> dict:
    """Median normalized response to isolated upward steps (single threshold crossings after silence)."""
    P, POST, show = idc["plateau_bins"], idc["post_bins"], idc["response_pre_bins"]
    resp = []
    for y in trials:
        d = np.diff(y, axis=0)                                  # d[i] = y[i+1] - y[i]
        n = d.shape[0]
        if n < P + POST + 1:
            continue
        i = np.arange(P, n - POST)                              # candidate step index
        pre = sliding_window_view(np.abs(d), P, axis=0).max(-1)[i - P]          # d[i-P:i]
        post = sliding_window_view(d, POST, axis=0).max(-1)[i + 1]              # d[i+1:i+1+POST]
        jump = d[i]
        ok = (jump > idc["jump_min"]) & (pre < idc["plateau_tol"] * jump) & (post < idc["post_tol"] * jump)
        ii, cc = np.nonzero(ok)
        if len(ii) == 0:
            continue
        t0 = i[ii] + 1
        ks = np.arange(-show, POST)
        seg = y[t0[:, None] + ks[None, :], cc[:, None]]
        resp.append((seg - y[t0 - 1, cc][:, None]) / jump[ii, cc][:, None])
    if not resp:
        return {"n_events": 0}
    r = np.concatenate(resp)
    med = np.median(r, 0)[show:]                                 # lags 0..POST-1
    neg = np.flatnonzero(med <= 0)
    length = int(neg[0]) if len(neg) else POST
    k = np.arange(length)
    # The crossing also enters the rolling statistics, which lowers every later z by a small delta (seen as
    # the negative level after the support). Inside the support the response is therefore
    # w_k / w_0 - delta * W_{k-1} (W = cumulative normalized weight); undo that before fitting.
    delta = -float(np.median(med[length:length + idc["after_support_bins"]])) if length < POST else 0.0
    target = med[:length] / med[0]
    for _ in range(20):
        cum = np.concatenate([[0.0], np.cumsum(target / target.sum())[:-1]])
        target = (med[:length] + delta * cum) / (med[0])
        target = target / target[0]

    def model(p):
        v = 1 / (1 + np.exp((k - p[0]) / p[1]))
        return v / v[0]
    fit = least_squares(lambda p: model(p) - target, x0=[length / 2, 1.0], bounds=([0, 1e-3], [np.inf, np.inf]))
    return {
        "n_events": int(r.shape[0]),
        "median_response_lag0_on": med,
        "p10_p90_at_lag5": np.quantile(r[:, show + 5], [0.1, 0.9]) if POST > 5 else None,
        "support_bins": length,
        "level_after_support": -delta if length < POST else None,
        "sigmoid_fit_center_bins": fit.x[0], "sigmoid_fit_scale_bins": fit.x[1],
        "sigmoid_fit_max_abs_resid": np.abs(fit.fun).max(),
    }


def deconvolve(trials: list, kernel: np.ndarray, burn: int) -> list:
    """Invert y = kernel * z (causal FIR) per trial; drop the first ``burn`` bins (unknown history)."""
    return [lfilter([1.0], kernel, y, axis=0)[burn:] for y in trials if y.shape[0] > burn]


def _levels(z: np.ndarray, idc: dict):
    """Cluster each (window x channel) block of deconvolved TX into count levels (sorted)."""
    W = idc["level_window_bins"]
    n_win = z.shape[0] // W
    s = np.sort(z[:n_win * W].reshape(n_win, W, -1), axis=1)
    lvl = np.concatenate([np.zeros((n_win, 1, s.shape[2]), int),
                          np.cumsum(np.diff(s, axis=1) > idc["level_gap"], axis=1)], axis=1)
    means, sizes = [], []
    for L in range(3):
        m = lvl == L
        sizes.append(m.sum(1))
        means.append(np.where(sizes[-1] > 0, (s * m).sum(1) / np.maximum(sizes[-1], 1), np.nan))
    return np.stack(means), np.stack(sizes), lvl.max(1)


def _ratio(off):
    return np.log((2 + off) / (1 + off)) / np.log((1 + off) / off)


def log_offset_from_levels(zs: list, idc: dict) -> dict:
    """Spacing of count levels n = 0, 1, 2 gives the log offset: (l2 - l1) / (l1 - l0) = ratio(offset)."""
    grid = np.logspace(*idc["offset_grid_log10"])
    est = []
    for z in zs:
        means, sizes, _ = _levels(z, idc)
        ok = np.all(sizes >= idc["min_level_size"], axis=0)
        rr = ((means[2] - means[1]) / (means[1] - means[0]))[ok]
        rr = rr[(rr > _ratio(grid[0])) & (rr < _ratio(grid[-1]))]
        est.append(np.interp(rr, _ratio(grid), np.log10(grid)))
    est = np.concatenate(est) if est else np.array([])
    if est.size == 0:
        return {"n_windows": 0}
    q = idc["offset_quantiles"]
    return {"n_windows": int(est.size), "log10_offset_quantiles": dict(zip(map(str, q), np.quantile(est, q)))}


def _counts(z: np.ndarray, idc: dict, offset: float) -> np.ndarray:
    """Integer counts per bin from the level positions (n = 0 lowest level, n = 1 next)."""
    W = idc["level_window_bins"]
    means, sizes, top = _levels(z, idc)
    n_win = means.shape[1]
    lv = np.log(np.arange(idc["max_count"] + 1) + offset)
    unit = (lv - lv[0]) / (lv[1] - lv[0])                       # level of n in units of (l1 - l0)
    zz = z[:n_win * W].reshape(n_win, W, -1)
    pred = means[0][:, None, :, None] + (means[1] - means[0])[:, None, :, None] * unit
    n = np.argmin(np.abs(zz[..., None] - pred), axis=-1)
    n = np.where((top == 0)[:, None, :], 0, n)                   # one level only: all zero
    return n.reshape(n_win * W, -1)


def rolling_hypotheses(zs: list, idc: dict, params: dict, bin_ms: float) -> dict:
    """RMSE between deconvolved TX z and the rolling z-score of log(recovered counts + offset)."""
    roll = params["rolling"]
    base_N = int(roll["window_ms"] // bin_ms)
    base_off = params["log_offset"]["threshcross"]
    hyps = [(N, inc, dd, base_off) for N in idc["windows_bins"] for inc in (True, False) for dd in (0, 1)]
    hyps += [(base_N, roll["include_current"], roll["ddof"], o) for o in idc["offsets"] if o != base_off]
    need = max(idc["windows_bins"]) + idc["long_trial_margin_bins"]
    err = {h: [] for h in hyps}
    n_used = 0
    for z in zs:
        if z.shape[0] < need:
            continue
        n = _counts(z, idc, base_off)
        T = n.shape[0]
        keep = n.sum(0) >= idc["min_events"]
        if not keep.any():
            continue
        n_used += 1
        for N, inc, dd, off in hyps:
            zz = rolling_zscore(np.log(n[:, keep] + off), N, inc, dd, roll["min_std"], roll["zero_std_value"])
            sl = slice(max(idc["windows_bins"]) + 1, T)
            err[(N, inc, dd, off)] += list(np.sqrt(np.mean((zz[sl] - z[:T][sl][:, keep]) ** 2, axis=0)))
    return {"n_long_trials": n_used, "n_channel_trials": len(err[hyps[0]]),
            "median_rmse": [{"window_bins": N, "include_current": inc, "ddof": dd, "log_offset": off,
                             "median_rmse": float(np.median(e)) if e else None}
                            for (N, inc, dd, off), e in err.items()]}


def deconvolved_summary(zs: list) -> dict:
    x = np.concatenate(zs)
    c = [z - z.mean(0) for z in zs]
    ac1 = np.sum([(a[1:] * a[:-1]).sum(0) for a in c], 0) / np.sum([(a * a).sum(0) for a in c], 0)
    return {"channel_mean_median": float(np.median(x.mean(0))), "channel_sd_median": float(np.median(x.std(0))),
            "channel_skew_median": float(np.median(skew(x, axis=0))), "acf1_median": float(np.median(ac1))}


def identify(data: dict, idc: dict, params: dict, bin_ms: float, rolling: bool) -> dict:
    sm = params["smoothing"]
    kernel = sigmoid_kernel(int(sm["length_ms"] // bin_ms), sm["center_ms"] / bin_ms, sm["scale_ms"] / bin_ms)
    tx_z = deconvolve(data["threshcross"], kernel, idc["burn_bins"])
    sp_z = deconvolve(data["spikepow"], kernel, idc["burn_bins"])
    out = {"tx_impulse_response": impulse_response(data["threshcross"], idc),
           "tx_log_offset_from_levels": log_offset_from_levels(tx_z, idc),
           "sbp_deconvolved": deconvolved_summary(sp_z), "tx_deconvolved": deconvolved_summary(tx_z)}
    if rolling:
        out["tx_rolling_hypotheses"] = rolling_hypotheses(tx_z, idc, params, bin_ms)
    return out


# ----------------------------------------------------------------------------- channel order

def channel_profiles(data: dict, idc: dict, params: dict, bin_ms: float) -> dict:
    """Per-channel TX rate (Hz, counts recovered from the processed features) and channel correlations."""
    sm = params["smoothing"]
    kernel = sigmoid_kernel(int(sm["length_ms"] // bin_ms), sm["center_ms"] / bin_ms, sm["scale_ms"] / bin_ms)
    tx_z = deconvolve(data["threshcross"], kernel, idc["burn_bins"])
    n = np.concatenate([_counts(z, idc, params["log_offset"]["threshcross"]) for z in tx_z])
    out = {"rate_hz": n.mean(0) * 1000.0 / bin_ms}
    for f in FEATS:
        out[f"corr_{f}"] = np.nan_to_num(np.corrcoef(np.concatenate(data[f]).T))
    return out


def reorder_permutation(mapping_path) -> np.ndarray:
    """Column index p with out[:, j] = in[:, p[j]], as their reorder_channels (mapping applied twice)."""
    m = np.asarray(json.loads(resolve_path(mapping_path).read_text())["electrode_mapping"], int) - 1
    return m[m]


def order_scores(kp: dict, wp: dict, perm: np.ndarray, cpa: int) -> dict:
    iu = np.triu_indices(len(perm), 1)
    out = {"rate_spearman": spearmanr(kp["rate_hz"][perm], wp["rate_hz"]).correlation}
    for f in FEATS:
        ck = kp[f"corr_{f}"][np.ix_(perm, perm)]
        out[f"corr_matrix_r_{f}"] = np.corrcoef(ck[iu], wp[f"corr_{f}"][iu])[0, 1]
    out["rate_spearman_per_array"] = [
        spearmanr(kp["rate_hz"][perm][a:a + cpa], wp["rate_hz"][a:a + cpa]).correlation
        for a in range(0, len(perm), cpa)]
    return out


def channel_order_check(kp: dict, wp: dict, co: dict, cpa: int, seed: int) -> dict:
    n = len(kp["rate_hz"])
    reo = reorder_permutation(co["mapping"])
    orders = {"as_is": np.arange(n), "reorder_channels": reo, "inverse_reorder_channels": np.argsort(reo)}
    res = {name: order_scores(kp, wp, p, cpa) for name, p in orders.items()}
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(co["n_null"]):
        p = np.concatenate([a + rng.permutation(cpa) for a in range(0, n, cpa)])
        null.append(order_scores(kp, wp, p, cpa))
    keys = ["rate_spearman"] + [f"corr_matrix_r_{f}" for f in FEATS]
    res["null_within_array_shuffle"] = {k: {"mean": np.mean([d[k] for d in null]),
                                            "p99": np.quantile([d[k] for d in null], 0.99)} for k in keys}
    res["reorder_channels_moves"] = {"channels": int((reo != np.arange(n)).sum()),
                                     "across_arrays": int((reo // cpa != np.arange(n) // cpa).sum())}
    return res


# ----------------------------------------------------------------------------- main

def _set_dotted(d: dict, key: str, value):
    *path, last = key.split(".")
    for p in path:
        d = d[p]
    d[last] = value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/kunz.yaml")
    args = ap.parse_args()
    set_seed(load_config()["seed"])
    cfg = load_yaml(resolve_path(args.config))
    adapter = cfg["adapter"]
    fc, bin_ms = adapter["feature_check"], adapter["bin_ms"]
    params = adapter["normalization"]["params"]
    n_ch = load_config()["model"]["n_features"] // 2
    result = {"config": {"normalization": adapter["normalization"], "clip_spikepow": adapter["clip_spikepow"],
                         "feature_check": fc},
              "comparison": {}, "identification": {}}

    w = load_wairagkar(fc)
    result["wairagkar_files"] = w["files"]
    print(f"Wairagkar: {len(w['spikepow'])} trials from {len(w['files'])} files")
    result["comparison"]["wairagkar_day482"] = {f: describe(w[f], fc, fc["wairagkar_arrays"]) for f in FEATS}
    result["identification"]["wairagkar_day482"] = identify(w, fc["identification"], params, bin_ms, rolling=True)
    del w

    co = fc["channel_order"]
    m195 = scipy.io.loadmat(resolve_path(co["wairagkar_file"]), variable_names=[f"{f}_trials" for f in FEATS])
    w195 = {f: [np.asarray(t, np.float64) for t in m195[f"{f}_trials"].ravel()] for f in FEATS}
    del m195
    print(f"Wairagkar {Path(co['wairagkar_file']).name}: {len(w195['spikepow'])} trials")
    wp = channel_profiles(w195, fc["identification"], params, bin_ms)
    del w195
    result["channel_order"] = {"wairagkar_file": co["wairagkar_file"], "mapping": co["mapping"]}

    kunz_arrays = None
    for cond in fc["conditions"]:
        m = scipy.io.loadmat(resolve_path(cfg["output_dir"]) / adapter["conditions"][cond]["file"],
                             variable_names=["chanSetNames"])
        kunz_arrays = [str(np.asarray(c).squeeze()) for c in m["chanSetNames"].ravel()]
        k = load_kunz(cfg, cond, adapter, n_ch)
        print(f"Kunz {cond}: {len(k['spikepow'])} trials")
        result["comparison"][f"kunz_{cond}"] = {f: describe(k[f], fc, kunz_arrays) for f in FEATS}
        result["identification"][f"kunz_{cond}_selfcheck"] = identify(k, fc["identification"], params, bin_ms,
                                                                      rolling=False)
        kp = channel_profiles(k, fc["identification"], params, bin_ms)
        del k
        raw = scipy.io.loadmat(resolve_path(cfg["output_dir"]) / adapter["conditions"][cond]["file"],
                               variable_names=["binnedTX", "goTrialEpochs"])
        raw_rate = np.concatenate([raw["binnedTX"][g0 - 1:g1] for g0, g1 in raw["goTrialEpochs"].astype(int)]
                                  ).mean(0) * 1000.0 / bin_ms
        del raw
        result["channel_order"][f"kunz_{cond}"] = channel_order_check(
            kp, wp, co, fc["channels_per_array"], load_config()["seed"])
        result["channel_order"][f"kunz_{cond}"]["rate_recovery_check"] = {
            "spearman_recovered_vs_raw_kunz_rate": spearmanr(kp["rate_hz"], raw_rate).correlation,
            "median_abs_rel_error": np.median(np.abs(kp["rate_hz"] - raw_rate) / np.maximum(raw_rate, 1e-9))}
        del kp
        result["comparison"][f"kunz_{cond}_sbp_zscore_no_log_no_smoothing"] = {
            "spikepow": describe(kunz_raw_sbp_zscore(cfg, cond, adapter), fc, kunz_arrays)}
        for vname, changes in fc["variants"].items():
            a = copy.deepcopy(adapter)
            for key, value in changes.items():
                _set_dotted(a["normalization"]["params"], key, value)
            k = load_kunz(cfg, cond, a, n_ch)
            result["comparison"][f"kunz_{cond}_{vname}"] = {"threshcross": describe(k["threshcross"], fc, kunz_arrays)}
            del k
    result["kunz_arrays"] = kunz_arrays

    out = resolve_path(fc["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_r(result), indent=1))
    print(f"wrote {fc['out']}")
    for name, comp in result["comparison"].items():
        for f, d in comp.items():
            print(f"{name:40s} {f:11s} mean {d['pooled']['mean']:+.3f} sd {d['pooled']['sd']:.3f} "
                  f"range [{d['pooled']['min']:+.2f}, {d['pooled']['max']:+.2f}] "
                  f"ch-sd median {d['channel_sd_quantiles']['0.5']:.3f} "
                  + " ".join(f"acf{lag} {v:.3f}" for lag, v in d["acf_median"].items()))
    for name, chk in result["channel_order"].items():
        if not isinstance(chk, dict):
            continue
        for order in ("as_is", "reorder_channels", "inverse_reorder_channels", "null_within_array_shuffle"):
            sc = chk[order]
            vals = {k: (v["mean"] if isinstance(v, dict) else v) for k, v in sc.items() if k != "rate_spearman_per_array"}
            print(f"channel order {name:16s} {order:26s} " + " ".join(f"{k} {v:+.3f}" for k, v in vals.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
