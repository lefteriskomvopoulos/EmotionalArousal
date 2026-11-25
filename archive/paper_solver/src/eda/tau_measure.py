"""
Measuring a participant's response shape tau directly from the signal, before any deconvolution.

The published solver cannot pin tau down: slower decay can always be traded for more pulses or
a bendier tonic, so it returns its built-in prior whatever the truth (tbme_score.rate_test,
window_test, tail_test).  With tau forced to the truth, far sessions improve (fixed_tau_test).
An isolated response has no neighbours to trade with, so its shape can be read directly:

  signal      100 Hz averaged to 10 Hz; stretches with gaps or artefacts (cohort_profile.jumps)
              are not used
  isolated    trough-to-peak rise >= 0.05 uS, and no other peak of at least 20% of its size
              from 3 s before its onset to 12 s after its peak
  fit         on that stretch: baseline line + a x (the model's response at onset t0), with
              t0, tau_rise, tau_decay free inside the authors' tau bounds, three starts
  kept        fit R2 >= 0.95, size > 0, tau off its bounds
  estimate    medians over a session's kept responses

Checks, all fixed before the run:
  synthetic (known tau), over sessions with >= 6 kept responses: median decay error <= 15%,
    rank correlation with the true decay >= 0.7, far vs prior sessions AUC >= 0.9.
  held out (real and synthetic): kept responses split odd/even in time, tau measured on one half
    and tried on the other, where each response keeps only t0, size and baseline free, so no
    pulses or tonic can hide a wrong shape.  Error ratio = median SSE(measured) / SSE(prior).
  stability: odd and even decays within 25% of each other.
  accepted: >= 6 kept responses, stable and error ratio <= 0.9; otherwise that participant
    keeps the prior, flagged.  Continuing to the cluster needs 3 of the 5 real participants.

A pass says the measured tau describes that person's isolated responses better than the prior,
not that it is their true tau; responses inside bursts may differ.

Saved: data/tbme_tau_responses.csv (every fitted response, kept or not)
       data/tbme_tau_measure.csv   (one row per session or participant)
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from . import cohort_profile as cp
from . import config as cfg
from . import tbme_export as tx
from . import tbme_synth as ts
from .dataset import load
from .tbme_score import auc

RESPONSES_CSV = cfg.ROOT / "data" / "tbme_tau_responses.csv"
MEASURE_CSV = cfg.ROOT / "data" / "tbme_tau_measure.csv"
REAL_IDS = (53, 115, 26, 3, 80)
FS = cfg.DFS
BLOCK = int(cfg.FS // FS)
MIN_RESP_uS = 0.05
OTHER_FRAC = 0.2
PRE_S, POST_S = 3.0, 12.0
T0_BACK_S = 1.5
RISE_BOUNDS, DECAY_BOUNDS = (0.1, 1.5), (1.5, 6.0)        # the authors' bounds
STARTS = ((0.65, 2.77), (0.35, 2.0), (1.2, 5.0))
MIN_FIT_R2 = 0.95
MIN_RESPONSES = 6
MAX_DECAY_ERR_PCT, MIN_RANK, MIN_AUC = 15, 0.7, 0.9
STABLE_RATIO, MAX_ERROR_RATIO = 1.25, 0.9
MIN_ACCEPTED_REAL = 3
PRIOR = (ts.PRIOR["rise"][0], ts.PRIOR["decay"][0])


def prepare(x, t):
    """100 Hz segment (NaN = missing) -> 10 Hz signal, times, mask of unusable samples."""
    xf, gaps = tx.fill(x, t)
    bad = gaps | cp.jumps(xf)[2]
    n = len(xf) // BLOCK
    block = lambda a: a[:n * BLOCK].reshape(n, BLOCK)
    return block(xf).mean(1), block(t).mean(1), block(bad).any(1)


def isolated(y, bad):
    """Isolated responses -> list of (first index, last index, onset, peak, size uS)."""
    pk, pr = find_peaks(y, prominence=cp.PEAK_MIN_uS, distance=max(int(cp.PEAK_GAP_S * FS), 1),
                        wlen=int(2 * cp.RISE_MAX_S * FS) + 1)
    out = []
    for k, (p, on) in enumerate(zip(pk, pr["left_bases"])):
        size = y[p] - y[on]
        i0, i1 = on - int(PRE_S * FS), p + int(POST_S * FS)
        if size < MIN_RESP_uS or i0 < 0 or i1 >= len(y) or bad[i0:i1 + 1].any():
            continue
        near = (pk >= i0) & (pk <= i1)
        near[k] = False
        if (pr["prominences"][near] >= OTHER_FRAC * size).any():
            continue
        out.append((i0, i1, on, p, size))
    return out


def shape(s, tau_rise, tau_decay):
    """The model's response s seconds after onset, peak 1 (tbme_synth.kernel, analytic)."""
    tau_decay = max(tau_decay, 1.01 * tau_rise)
    tp = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    s = np.maximum(s, 0)
    return (np.exp(-s / tau_decay) - np.exp(-s / tau_rise)) / (
        np.exp(-tp / tau_decay) - np.exp(-tp / tau_rise))


def residual(t, y, t0, tau_rise, tau_decay):
    """Residual after the best baseline line + size for this shape -> (residual, size)."""
    X = np.column_stack([np.ones_like(t), t - t.mean(), shape(t - t0, tau_rise, tau_decay)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta, beta[2]


def fit_free(t, y, t_on, t_pk):
    """t0, tau_rise, tau_decay free -> dict for the best of the starts."""
    lo = [t_on - T0_BACK_S, RISE_BOUNDS[0], DECAY_BOUNDS[0]]
    hi = [t_pk, RISE_BOUNDS[1], DECAY_BOUNDS[1]]
    best = None
    for tr, td in STARTS:
        r = least_squares(lambda p: residual(t, y, *p)[0], [t_on, tr, td], bounds=(lo, hi),
                          x_scale=[0.5, 0.2, 1.0])
        if best is None or r.cost < best.cost:
            best = r
    t0, tr, td = best.x
    res, size = residual(t, y, t0, tr, td)
    on_bound = any(min(abs(v - a), abs(v - b)) < 0.01 * v for v, (a, b) in
                   ((tr, RISE_BOUNDS), (td, DECAY_BOUNDS)))
    r2 = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)
    return {"tau_rise": tr, "tau_decay": td, "t0": t0, "fit_size": size, "fit_r2": r2,
            "on_bound": on_bound, "kept": bool(size > 0 and r2 >= MIN_FIT_R2 and not on_bound)}


def sse_fixed(t, y, t_on, t_pk, tau_rise, tau_decay):
    """Squared error of a response with tau given; only t0, size and baseline are free."""
    r = least_squares(lambda p: residual(t, y, p[0], tau_rise, tau_decay)[0], [t_on],
                      bounds=([t_on - T0_BACK_S], [t_pk]), x_scale=[0.5])
    return 2 * r.cost


def measure_session(name, kind, x, t):
    """One session -> (one row per fitted response, one summary dict)."""
    y, tt, bad = prepare(x, t)
    found = isolated(y, bad)
    rows = []
    for i0, i1, on, p, size in found:
        seg_t, seg_y = tt[i0:i1 + 1], y[i0:i1 + 1]
        rows.append({"name": name, "kind": kind, "time_s": tt[p], "size_uS": size,
                     **fit_free(seg_t, seg_y, tt[on], tt[p]),
                     "_seg": (seg_t, seg_y, tt[on], tt[p])})
    kept = [r for r in rows if r["kept"]]
    for k, r in enumerate(kept):
        r["half"] = k % 2
    out = {"name": name, "kind": kind, "n_isolated": len(rows), "n_kept": len(kept),
           "tau_rise_measured": np.median([r["tau_rise"] for r in kept]) if kept else np.nan,
           "tau_decay_measured": np.median([r["tau_decay"] for r in kept]) if kept else np.nan,
           "decay_odd": np.nan, "decay_even": np.nan, "stable": False, "error_ratio": np.nan,
           "share_better": np.nan, "accepted": False}
    if len(kept) >= MIN_RESPONSES:
        med = {h: (np.median([r["tau_rise"] for r in kept if r["half"] == h]),
                   np.median([r["tau_decay"] for r in kept if r["half"] == h])) for h in (0, 1)}
        for r in kept:
            r["sse_measured"] = sse_fixed(*r["_seg"], *med[1 - r["half"]])
            r["sse_prior"] = sse_fixed(*r["_seg"], *PRIOR)
        ratio = np.array([r["sse_measured"] / r["sse_prior"] for r in kept])
        a, b = med[0][1], med[1][1]
        out.update(decay_odd=a, decay_even=b, stable=bool(max(a, b) / min(a, b) <= STABLE_RATIO),
                   error_ratio=np.median(ratio), share_better=(ratio < 1).mean())
        out["accepted"] = bool(out["stable"] and out["error_ratio"] <= MAX_ERROR_RATIO)
    for r in rows:
        del r["_seg"]
    return rows, out


def run(real_ids=REAL_IDS):
    """Step 2 (36 synthetic sessions, known tau) and step 3 (real participants)."""
    rows, summary = [], []
    for name, row, group, tau_rise, tau_decay, t, mk, s in ts.rebuilt():
        rs, out = measure_session(name, "synthetic", s["x"], t)
        rows += rs
        summary.append({**out, "tau_group": group, "tau_rise_true": tau_rise,
                        "tau_decay_true": tau_decay})
    for pid in real_ids:
        d = load(pid)
        sl = tx.segment(d)
        rs, out = measure_session(f"P{pid}", "real", d["eda"][sl], d["t"][sl])
        rows += rs
        summary.append({**out, "tau_group": "real"})
    pd.DataFrame(rows).to_csv(RESPONSES_CSV, index=False)
    res = pd.DataFrame(summary)
    res.to_csv(MEASURE_CSV, index=False)

    syn = res[res.kind == "synthetic"].copy()
    enough = syn[syn.n_kept >= MIN_RESPONSES].copy()
    enough["decay_err_pct"] = 100 * (enough.tau_decay_measured / enough.tau_decay_true - 1)
    enough["rise_err_pct"] = 100 * (enough.tau_rise_measured / enough.tau_rise_true - 1)
    print("synthetic, median by tau group:")
    print(syn.groupby("tau_group").agg(sessions=("name", "size"), n_kept=("n_kept", "median"),
                                       enough=("n_kept", lambda v: (v >= MIN_RESPONSES).sum())).join(
        enough.groupby("tau_group")[["tau_decay_true", "tau_decay_measured", "decay_err_pct",
                                     "tau_rise_true", "tau_rise_measured", "rise_err_pct",
                                     "error_ratio", "accepted"]].median())
          .to_string(float_format=lambda v: f"{v:.3f}"))
    err = enough.decay_err_pct.abs().median()
    rank = spearmanr(enough.tau_decay_true, enough.tau_decay_measured)[0]
    area = auc(enough.tau_decay_measured[enough.tau_group == "far"],
               enough.tau_decay_measured[enough.tau_group == "prior"])
    passed = err <= MAX_DECAY_ERR_PCT and rank >= MIN_RANK and area >= MIN_AUC
    print(f"step 2, {len(enough)}/{len(syn)} sessions usable: median decay error {err:.1f}% "
          f"(<= {MAX_DECAY_ERR_PCT}), rank correlation {rank:.2f} (>= {MIN_RANK}), far vs prior AUC "
          f"{area:.2f} (>= {MIN_AUC}) -> {'PASS' if passed else 'FAIL'}")
    print(f"       rise: median error {enough.rise_err_pct.abs().median():.1f}%, rank correlation "
          f"{spearmanr(enough.tau_rise_true, enough.tau_rise_measured)[0]:.2f} (not in the rule)")
    real = res[res.kind == "real"]
    print("real participants:")
    print(real[["name", "n_isolated", "n_kept", "tau_rise_measured", "tau_decay_measured",
                "decay_odd", "decay_even", "stable", "error_ratio", "share_better", "accepted"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"step 3: {real.accepted.sum()}/{len(real)} accepted (>= {MIN_ACCEPTED_REAL} to continue) "
          f"-> {'PASS' if real.accepted.sum() >= MIN_ACCEPTED_REAL else 'FAIL'}")
    print("saved", RESPONSES_CSV.name, MEASURE_CSV.name)
    return res
