"""
One shared response shape per person, measured from all their isolated responses.

Post-hoc retry of tau_measure.py, which passed on synthetic sessions but failed on real ones (0 of
5 accepted): per-response fits hit the limits, so few were kept.  Changed after seeing that result,
to be disclosed: one tau for all of a person's responses, wider limits, no per-response filtering,
and the signal the solver now receives.  The acceptance rules are unchanged, plus one: a tau on
the edge of the limits is refused.

  signal      paper's preprocessing (tbme_export.paper_preprocess), averaged to 10 Hz; stretches
              with gaps or artefacts in the unfiltered signal are not used
  isolated    tau_measure.isolated, same rules
  error       per response, baseline line and size solved, onset t0 searched: squared error as a
              share of the response's variance around that line, 0 to 1, so no response dominates
  estimate    the (tau_rise, tau_decay) on a grid with the lowest mean error over the responses
  limits      tau_rise 0.1-3 s, tau_decay 1.5-12 s

Checks: same as tau_measure.  Synthetic sessions with >= 6 responses must give median decay error
<= 15%, rank correlation >= 0.7 and far vs prior AUC >= 0.9.  Each person's responses are split
odd/even in time, tau is measured on one half and scored on the other against the prior.
Accepted: >= 6 responses, odd and even decay within 25%, error ratio <= 0.9, tau off the edges.
Continuing to the cluster needs 3 of the 5 real participants.

A pass says this tau describes that person's isolated responses, as filtered, better than the
prior; it is not proof of their true tau.  Filtering stretches the rise, so tau_rise is expected
to read above the unfiltered truth.

Saved: data/tbme_tau_shared_responses.csv (per response, held-out errors)
       data/tbme_tau_shared.csv           (one row per session or participant)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import cohort_profile as cp
from . import config as cfg
from . import tau_measure as tm
from . import tbme_export as tx
from . import tbme_synth as ts
from .dataset import load
from .tbme_score import auc

RESPONSES_CSV = cfg.ROOT / "data" / "tbme_tau_shared_responses.csv"
SHARED_CSV = cfg.ROOT / "data" / "tbme_tau_shared.csv"
RISE_GRID = np.round(np.arange(0.1, 3.001, 0.1), 2)
DECAY_GRID = np.round(np.arange(1.5, 12.001, 0.25), 2)
T0_STEP_S = 0.1


def prepare(x, t):
    """100 Hz segment -> preprocessed 10 Hz signal, times, mask of unusable samples."""
    xf, gaps = tx.fill(x, t)
    bad = gaps | cp.jumps(xf)[2]
    y, _ = tx.paper_preprocess(x, t)
    n = len(y) // tm.BLOCK
    block = lambda a: a[:n * tm.BLOCK].reshape(n, tm.BLOCK)
    return block(y).mean(1), block(t).mean(1), block(bad).any(1)


def stretch(t, y, t_on, t_pk):
    """One response, ready for repeated scoring: line basis, detrended signal, onset lags."""
    q = np.linalg.qr(np.column_stack([np.ones_like(t), t - t.mean()]))[0]
    t0 = np.arange(t_on - tm.T0_BACK_S, t_pk + 1e-9, T0_STEP_S)
    return q, y - q @ (q.T @ y), t[:, None] - t0[None, :]


def rel_error(seg, tau_rise, tau_decay):
    """Share of the response's variance left unexplained, at its best onset; 1 = nothing fitted."""
    q, yp, lag = seg
    s = tm.shape(lag, tau_rise, tau_decay)
    s = s - q @ (q.T @ s)
    num, den = s.T @ yp, np.sum(s * s, 0)
    gain = np.where((num > 0) & (den > 0), num ** 2 / np.maximum(den, 1e-30), 0.0)
    return 1 - gain.max() / np.sum(yp ** 2)


def error_grid(segs):
    """(responses, rise grid, decay grid) array of errors; NaN where decay <= rise."""
    e = np.full((len(segs), len(RISE_GRID), len(DECAY_GRID)), np.nan)
    for i, tr in enumerate(RISE_GRID):
        for j, td in enumerate(DECAY_GRID):
            if td > tr:
                e[:, i, j] = [rel_error(seg, tr, td) for seg in segs]
    return e


def best_tau(e):
    """Lowest mean error over the responses in e -> (rise index, decay index, on the edge?)."""
    i, j = np.unravel_index(np.nanargmin(e.mean(0)), e.shape[1:])
    return i, j, bool(i in (0, len(RISE_GRID) - 1) or j in (0, len(DECAY_GRID) - 1))


def measure_session(name, kind, x, t):
    """One session -> (one row per response, one summary dict)."""
    y, tt, bad = prepare(x, t)
    found = tm.isolated(y, bad)
    segs = [stretch(tt[i0:i1 + 1], y[i0:i1 + 1], tt[on], tt[p]) for i0, i1, on, p, _ in found]
    rows = [{"name": name, "kind": kind, "time_s": tt[p], "size_uS": size, "half": k % 2}
            for k, (_, _, _, p, size) in enumerate(found)]
    out = {"name": name, "kind": kind, "n_isolated": len(found), "tau_rise_shared": np.nan,
           "tau_decay_shared": np.nan, "on_edge": False, "error_shared": np.nan,
           "error_prior": np.nan, "decay_odd": np.nan, "decay_even": np.nan, "stable": False,
           "error_ratio": np.nan, "share_better": np.nan, "accepted": False}
    if len(found) < tm.MIN_RESPONSES:
        return rows, out
    e = error_grid(segs)
    prior = np.array([rel_error(seg, *tm.PRIOR) for seg in segs])
    i, j, edge = best_tau(e)
    half = np.arange(len(segs)) % 2
    held = np.empty(len(segs))
    decay = {}
    for h in (0, 1):
        a, b, _ = best_tau(e[half == h])
        decay[h] = DECAY_GRID[b]
        held[half != h] = e[half != h, a, b]
    ratio = held / np.maximum(prior, 1e-30)
    for r, m, p, q in zip(rows, held, prior, ratio):
        r.update(error_held_out=m, error_prior=p, error_ratio=q)
    stable = max(decay.values()) / min(decay.values()) <= tm.STABLE_RATIO
    out.update(tau_rise_shared=RISE_GRID[i], tau_decay_shared=DECAY_GRID[j], on_edge=edge,
               error_shared=e.mean(0)[i, j], error_prior=prior.mean(), decay_odd=decay[0],
               decay_even=decay[1], stable=bool(stable), error_ratio=np.median(ratio),
               share_better=(ratio < 1).mean())
    out["accepted"] = bool(stable and out["error_ratio"] <= tm.MAX_ERROR_RATIO and not edge)
    return rows, out


def run(real_ids=tm.REAL_IDS):
    """36 synthetic sessions (known tau, sizes capped as in S<k>_pp) and the real participants."""
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    rows, summary = [], []
    for name, row, group, tau_rise, tau_decay, t, mk, s in ts.rebuilt(amp_caps=caps):
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
    res.to_csv(SHARED_CSV, index=False)

    syn = res[res.kind == "synthetic"]
    enough = syn[syn.n_isolated >= tm.MIN_RESPONSES].copy()
    enough["decay_err_pct"] = 100 * (enough.tau_decay_shared / enough.tau_decay_true - 1)
    enough["rise_err_pct"] = 100 * (enough.tau_rise_shared / enough.tau_rise_true - 1)
    print("synthetic, median by tau group:")
    print(enough.groupby("tau_group").agg(
        sessions=("name", "size"), n_isolated=("n_isolated", "median"),
        decay_true=("tau_decay_true", "median"), decay_shared=("tau_decay_shared", "median"),
        decay_err_pct=("decay_err_pct", "median"), rise_true=("tau_rise_true", "median"),
        rise_shared=("tau_rise_shared", "median"), error_ratio=("error_ratio", "median"),
        accepted=("accepted", "mean"), on_edge=("on_edge", "mean"))
        .to_string(float_format=lambda v: f"{v:.3f}"))
    err = enough.decay_err_pct.abs().median()
    rank = spearmanr(enough.tau_decay_true, enough.tau_decay_shared)[0]
    area = auc(enough.tau_decay_shared[enough.tau_group == "far"],
               enough.tau_decay_shared[enough.tau_group == "prior"])
    passed = err <= tm.MAX_DECAY_ERR_PCT and rank >= tm.MIN_RANK and area >= tm.MIN_AUC
    print(f"step 2, {len(enough)}/{len(syn)} sessions usable: median decay error {err:.1f}% "
          f"(<= {tm.MAX_DECAY_ERR_PCT}), rank correlation {rank:.2f} (>= {tm.MIN_RANK}), far vs "
          f"prior AUC {area:.2f} (>= {tm.MIN_AUC}) -> {'PASS' if passed else 'FAIL'}")
    print(f"       rise: median error {enough.rise_err_pct.abs().median():.1f}%, rank correlation "
          f"{spearmanr(enough.tau_rise_true, enough.tau_rise_shared)[0]:.2f} (not in the rule)")
    real = res[res.kind == "real"]
    print("real participants:")
    print(real[["name", "n_isolated", "tau_rise_shared", "tau_decay_shared", "on_edge", "decay_odd",
                "decay_even", "stable", "error_ratio", "share_better", "accepted"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"step 3: {real.accepted.sum()}/{len(real)} accepted (>= {tm.MIN_ACCEPTED_REAL} to "
          f"continue) -> {'PASS' if passed and real.accepted.sum() >= tm.MIN_ACCEPTED_REAL else 'FAIL'}")
    print("saved", RESPONSES_CSV.name, SHARED_CSV.name)
    return res
