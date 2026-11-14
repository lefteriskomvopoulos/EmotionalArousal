"""
Model-free SCR shape measurement (scr-shapes): rise and recovery times of isolated SCRs, measured
on the waveform as in classic SCR scoring (Boucsein; Dawson et al.), with no model output.

  signal     each recording through prepare() (jumps repaired) with a light SHAPE_CUTOFF_HZ low-pass
             at SHAPE_FS: gentler than the solver's 1 Hz, so rise times are not smeared
  SCRs       peaks of prominence >= BIG_uS (clearly above the noise)
  onset      the lowest point in the ONSET_WIN_S before the peak; dropped if that is the window's
             first sample (no minimum inside the window)
  baseline   a straight line fitted to the BASE_S before onset and extended (removes tonic drift
             and the tail of an earlier response); amplitude = peak above that line, dropped if
             under BIG_uS
  half-rise  the 50% point on the way up to the peak (last crossing before it), linearly
             interpolated between samples
  half       peak to the first 50% point on the way down, interpolated, within MAX_HALF_S
             (otherwise dropped).  Both times sit where the signal is steep, so noise moves them
             little; the onset itself sits in a flat, noisy stretch (first version: onset to peak
             came out 43% long on the synthetic check)
  isolated   no other peak of prominence >= 0.02 uS from ISOLATE_BEFORE_S before onset to the
             half-recovery point
  shape      each SCR's (half-rise, half) matched to the (tau_rise, tau_decay) of the solver's
             response kernel with the same two times (shape_grid, within the solver's bounds)

Run on the 50 selected participants (the population), compared per participant with the solver's
fits on them (final run) in the same units, and checked on the 50 synthetic test sessions, whose
true shapes are known.  Pass criterion, set before the run: on the sessions with >= MIN_SCRS
isolated SCRs, median error within +-MAX_ERR_PCT and Spearman >= MIN_RHO (measured vs true across
sessions), for half-recovery and decay.  A measurement that fails it is not used as a prior.
Saved: data/scr_shapes_real50.csv (every isolated SCR), data/prior_measured50.csv (the summary),
data/scr_shapes_validation.csv (the synthetic check).

Result (2026-11-12, final; not refined further): half-rise PASS (-0.1%, Spearman .97); recovery
+12.9% (.93) and decay +13.9% (.91) FAIL, so no measured prior.  Real 50: 444 isolated SCRs, half-rise
1.07 s, recovery 2.49 s; 36% recover as fast as they rise (decay at the 1.5 s grid floor).  Per
participant the solver's fits run longer (half-rise x1.10, recovery x1.19, Spearman .61 / .58).
Used as a description only.
"""
from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from . import cohort_profile as cp
from . import config as cfg
from . import generator as gen
from . import preprocess as pp
from . import runs
from . import tbme_export as tx
from . import tbme_synth as ts
from .dataset import load

SHAPE_CUTOFF_HZ = 3.0
SHAPE_FS = 20
BIG_uS = 0.05
ONSET_WIN_S = 5.0
ISOLATE_BEFORE_S = 5.0
BASE_S = 3.0
MAX_HALF_S = 20.0
MIN_SCRS = 5
MAX_ERR_PCT = 10.0
MIN_RHO = 0.8
COLS = ["i_peak", "amp_uS", "half_rise_s", "half_s", "onset_rise_s"]
REAL_CSV = cfg.ROOT / "data" / "scr_shapes_real50.csv"
PRIOR_CSV = cfg.ROOT / "data" / "prior_measured50.csv"
VALID_CSV = cfg.ROOT / "data" / "scr_shapes_validation.csv"


def measure(y, fs=SHAPE_FS):
    """Isolated SCRs of one signal -> (DataFrame of COLS, Counter of dropped peaks by reason)."""
    big = find_peaks(y, prominence=BIG_uS, distance=int(fs))[0]
    every = find_peaks(y, prominence=cfg.MIN_EVENT_uS, distance=int(fs))[0]
    rows, drop = [], Counter()
    for p in big:
        a = p - int(ONSET_WIN_S * fs)
        b0 = a - int(BASE_S * fs)
        if b0 < 0:
            drop["at the start"] += 1
            continue
        on = a + int(np.argmin(y[a:p + 1]))
        if on == a or on == p:
            drop["no onset in window"] += 1
            continue
        b = on - int(BASE_S * fs)
        slope = np.polyfit(np.arange(b, on + 1), y[b:on + 1], 1)[0]
        k = np.arange(on, min(p + int(MAX_HALF_S * fs), len(y) - 1) + 1)
        z = y[k] - (y[on] + slope * (k - on))           # above the extended baseline, z[0] = 0
        q = p - on
        amp = z[q]
        if amp < BIG_uS:
            drop["small above baseline"] += 1
            continue
        lev = 0.5 * amp
        i = np.flatnonzero(z[:q] < lev)[-1]              # last sample under half before the peak
        up = i + (lev - z[i]) / (z[i + 1] - z[i])
        after = np.flatnonzero(z[q:] <= lev)
        if not len(after):
            drop["no half-recovery in 20 s"] += 1
            continue
        j = q + after[0]                                 # first sample at or under half after it
        down = j - 1 + (z[j - 1] - lev) / (z[j - 1] - z[j])
        near = every[(every != p) & (every >= on - int(ISOLATE_BEFORE_S * fs)) & (every <= on + j)]
        if len(near):
            drop["not isolated"] += 1
            continue
        rows.append({"i_peak": int(p), "amp_uS": float(amp), "half_rise_s": (q - up) / fs,
                     "half_s": (down - q) / fs, "onset_rise_s": q / fs})
    drop["kept"] = len(rows)
    return pd.DataFrame(rows, columns=COLS), drop


def half_times(tau_rise, tau_decay):
    """Kernel exp(-t/td) - exp(-t/tr), elementwise: (onset to peak, 50% up to peak, peak to 50%
    down), seconds, by bisection."""
    tr = np.atleast_1d(np.asarray(tau_rise, float))
    td = np.atleast_1d(np.asarray(tau_decay, float))
    tp = ts.tp_of(tr, td)
    h = lambda t: np.exp(-t / td) - np.exp(-t / tr)
    level = 0.5 * h(tp)
    lo, hi = np.zeros_like(tp), tp.copy()                # rising side
    for _ in range(60):
        mid = (lo + hi) / 2
        under = h(mid) < level
        lo, hi = np.where(under, mid, lo), np.where(under, hi, mid)
    rise = tp - (lo + hi) / 2
    lo, hi = tp.copy(), tp + 20 * td                     # falling side
    for _ in range(60):
        mid = (lo + hi) / 2
        over = h(mid) > level
        lo, hi = np.where(over, mid, lo), np.where(over, hi, mid)
    return tp, rise, (lo + hi) / 2 - tp


def shape_grid():
    """The solver's response kernel over its bounds: tau_rise 0.1-2.0 s, tau_decay 1.5-14 s,
    decay >= 1.1 x rise -> (tau_rise, tau_decay, onset to peak, half-rise, half-recovery)."""
    tr, td = np.meshgrid(np.arange(0.1, 2.0001, 0.01), np.arange(1.5, 14.0001, 0.05), indexing="ij")
    tr, td = tr.ravel(), td.ravel()
    keep = td >= 1.1 * tr
    tr, td = tr[keep], td[keep]
    return (tr, td) + half_times(tr, td)


def to_tau(half_rise, half, grid):
    """(tau_rise, tau_decay, onset to peak, relative misfit) of the grid kernel nearest a measured
    (half-rise, half-recovery)."""
    tr, td, tp, hr, hh = grid
    err = ((hr - half_rise) / half_rise) ** 2 + ((hh - half) / half) ** 2
    j = int(np.argmin(err))
    return float(tr[j]), float(td[j]), float(tp[j]), float(np.sqrt(err[j]))


def _with_tau(m, grid):
    if not len(m):
        return m.assign(tau_rise=[], tau_decay=[], tp_s=[], misfit=[])
    v = np.array([to_tau(r, h, grid) for r, h in zip(m.half_rise_s, m.half_s)])
    return m.assign(tau_rise=v[:, 0], tau_decay=v[:, 1], tp_s=v[:, 2], misfit=v[:, 3])


def robust(v):
    """(median, robust SD = 1.4826 x median absolute deviation)."""
    v = np.asarray(v, float)
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med)))


def _drops(c):
    total = sum(c.values())
    return ", ".join(f"{k} {v} ({v / total:.0%})" for k, v in c.most_common())


def run(suffix="own4"):
    """scr-shapes: measure the 50 real participants, compare with the solver's fits, check on the
    synthetic test sessions."""
    grid = shape_grid()
    ids = gen.selected_ids()
    measures = ("half_rise_s", "half_s", "tau_rise", "tau_decay")

    rows, drops = [], Counter()
    for k, pid in enumerate(ids, 1):
        d = load(pid, every=1)
        p = pp.prepare(d["eda"], d["t"], pp.RAW_FS, pp.segment_of(d["marker"]), SHAPE_CUTOFF_HZ, SHAPE_FS)
        m, c = measure(p["y"])
        drops += c
        if len(m):
            m = _with_tau(m, grid)
            rows.append(m.assign(source_id=pid, t_peak=p["t2"][m.i_peak.to_numpy(int)]))
        if k % 10 == 0:
            print(f"  real: {k}/{len(ids)} participants")
    real = pd.concat(rows, ignore_index=True)
    real.to_csv(REAL_CSV, index=False)

    per = real.groupby("source_id")[list(measures)].median()
    per["n"] = real.groupby("source_id").size()
    sol = {}
    for pid in ids:
        w = pd.read_csv(runs.FINAL_OUT / runs.FINAL6_SETTING / f"P{pid}_{suffix}_windows.csv")
        w = w[(w.selection == "authors") & (w.tau_decay > w.tau_rise)]
        _, hr, hh = half_times(w.tau_rise.to_numpy(), w.tau_decay.to_numpy())
        sol[pid] = {"tau_rise": w.tau_rise.median(), "tau_decay": w.tau_decay.median(),
                    "half_rise_s": np.median(hr), "half_s": np.median(hh)}
    for a in measures:
        per[f"solver_{a}"] = [sol[i][a] for i in per.index]

    summary = []
    for col in measures + ("tp_s",):
        c, s = robust(real[col])
        summary.append({"measure": col, "median": c, "robust_sd": s, "q10": real[col].quantile(.1),
                        "q90": real[col].quantile(.9), "n_scrs": len(real), "n_participants": real.source_id.nunique()})
    summ = pd.DataFrame(summary)
    summ.to_csv(PRIOR_CSV, index=False)

    f = lambda v: f"{v:.3f}"
    at_bound = (np.isclose(real.tau_rise, 0.1) | np.isclose(real.tau_rise, 2.0)
                | np.isclose(real.tau_decay, 1.5) | np.isclose(real.tau_decay, 14.0)).mean()
    print(f"\nbig peaks (>= {BIG_uS} uS): {_drops(drops)}")
    print(f"isolated SCRs: {len(real)} from {real.source_id.nunique()} of {len(ids)} participants"
          f" (per participant median {per.n.median():.0f}, {per.n.min()}-{per.n.max()})")
    print(summ.to_string(index=False, float_format=f))
    print(f"literature: rise (onset to peak) 1-3 s, half-recovery 2-10 s; within: rise {real.tp_s.between(1, 3).mean():.0%},"
          f" half {real.half_s.between(2, 10).mean():.0%}; grid misfit median {real.misfit.median():.3f},"
          f" at the grid's bounds {at_bound:.0%}")
    ok = per[per.n >= MIN_SCRS]
    print(f"\nper participant with >= {MIN_SCRS} SCRs (n {len(ok)}), measured (median of their SCRs) vs the"
          f" solver's fits (final run, median over windows):")
    for a in measures:
        print(f"  {a}: measured {ok[a].median():.2f} s, solver {ok[f'solver_{a}'].median():.2f} s, ratio solver /"
              f" measured {np.median(ok[f'solver_{a}'] / ok[a]):.2f}, Spearman {spearmanr(ok[a], ok[f'solver_{a}'])[0]:.2f}")

    # the check: the same measurement on the synthetic test sessions (known shapes), rebuilt at 100 Hz
    # exactly as test50-export made them (same draw, same seeds)
    prof = pd.read_csv(cp.PROFILE_CSV)
    picked = prof[prof.id.isin(ids)].sort_values("scr_pm")
    curves = np.load(cp.CURVES_NPZ)
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    rp = gen.real_profiles(ids)
    rng = np.random.default_rng(gen.TEST6_SEED)
    rows, drops = [], Counter()
    for k, row in enumerate(picked.itertuples(), 1):
        pid = int(row.id)
        tr, td, g = gen.draw_shape(rng)
        target = pp.noise_2hz(loadmat(tx.OUT / f"P{pid}_{suffix}.mat")["y"].ravel())
        s0 = gen.TEST6_SEED
        t, mk, s, x = gen.flat_session(row, rp[pid], tr, td, (s0 + pid, s0 + 200000 + pid, s0 + 300000 + pid),
                                      target, curves, caps, suffix)
        p = pp.prepare(x, t, cfg.FS, (0, len(t) - 1), SHAPE_CUTOFF_HZ, SHAPE_FS)
        m, c = measure(p["y"])
        drops += c
        m = _with_tau(m, grid)
        _, hr, hh = (float(v[0]) for v in half_times(tr, td))
        r = {"session": f"V{pid}", "shape": g, "n": len(m), "true_tau_rise": tr, "true_tau_decay": td,
             "true_half_rise_s": hr, "true_half_s": hh}
        r.update({a: m[a].median() if len(m) else np.nan for a in measures})
        rows.append(r)
        if k % 10 == 0:
            print(f"  check: {k}/{len(ids)} test sessions")
    v = pd.DataFrame(rows)
    for a in measures:
        v[f"{a}_err_pct"] = 100 * (v[a] / v[f"true_{a}"] - 1)
    v.to_csv(VALID_CSV, index=False)
    cols = [f"{a}_err_pct" for a in measures]
    use = v[v.n >= MIN_SCRS]
    t = use.groupby("shape")[cols + ["n"]].median()
    t.loc["all"] = use[cols + ["n"]].median()
    print(f"\ncheck on the synthetic test sessions (true shape known), big peaks: {_drops(drops)}")
    print(f"sessions with >= {MIN_SCRS} isolated SCRs: {len(use)} of {len(v)}; measured / true - 1, %, median")
    print(t.to_string(float_format=lambda x: f"{x:.1f}"))
    print("  10-90%:", ", ".join(f"{a} {use[c].quantile(.1):.0f} to {use[c].quantile(.9):.0f}" for a, c in zip(measures, cols)))
    print(f"criterion (set before the run): median error within +-{MAX_ERR_PCT:.0f}% and Spearman >= {MIN_RHO}")
    for a, c in zip(measures, cols):
        err, rho = use[c].median(), spearmanr(use[a], use[f"true_{a}"])[0]
        verdict = "PASS" if abs(err) <= MAX_ERR_PCT and rho >= MIN_RHO else "FAIL"
        need = " (required)" if a in ("half_s", "tau_decay") else ""
        print(f"  {a}: error {err:+.1f}%, Spearman {rho:.2f} -> {verdict}{need}")
    print("saved", REAL_CSV.name, PRIOR_CSV.name, VALID_CSV.name)
    return real, summ, v


PRE_S = 1.0       # pulses up to this long before an SCR's measured onset still belong to it (the onset is
                  # the lowest point before the peak, which sits early, not late)
PULSE_CHECK_CSV = cfg.ROOT / "data" / "pulse_check_isolated.csv"


def _test_sessions(ids, suffix="own4"):
    """The 50 test sessions rebuilt at 100 Hz exactly as test50-export made them and as run() checks
    them (same draw, same seeds), each measured -> (pid, shape, isolated SCRs with t_peak)."""
    prof = pd.read_csv(cp.PROFILE_CSV)
    picked = prof[prof.id.isin(ids)].sort_values("scr_pm")
    curves = np.load(cp.CURVES_NPZ)
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    rp = gen.real_profiles(ids)
    rng = np.random.default_rng(gen.TEST6_SEED)
    for k, row in enumerate(picked.itertuples(), 1):
        pid = int(row.id)
        tr, td, g = gen.draw_shape(rng)
        target = pp.noise_2hz(loadmat(tx.OUT / f"P{pid}_{suffix}.mat")["y"].ravel())
        s0 = gen.TEST6_SEED
        t, mk, s, x = gen.flat_session(row, rp[pid], tr, td, (s0 + pid, s0 + 200000 + pid, s0 + 300000 + pid),
                                      target, curves, caps, suffix)
        p = pp.prepare(x, t, cfg.FS, (0, len(t) - 1), SHAPE_CUTOFF_HZ, SHAPE_FS)
        m, _ = measure(p["y"])
        if len(m):
            m = m.assign(t_peak=p["t2"][m.i_peak.to_numpy(int)])
        if k % 10 == 0:
            print(f"  test sessions: {k}/{len(ids)} rebuilt and measured")
        yield pid, g, m


def _in_windows(m, times, heights=None, min_h=None):
    """Pulses in each SCR's rise, from PRE_S before its onset to its peak, and in its recovery, from
    the peak to the half-recovery point.  Only pulses of height >= min_h when given."""
    t = np.asarray(times, float)
    if min_h is not None:
        t = t[np.asarray(heights, float) >= min_h]
    t = np.sort(t)
    pk = m.t_peak.to_numpy(float)
    on, hf = pk - m.onset_rise_s.to_numpy(float), pk + m.half_s.to_numpy(float)
    rise = np.searchsorted(t, pk, side="right") - np.searchsorted(t, on - PRE_S, side="left")
    tail = np.searchsorted(t, hf, side="right") - np.searchsorted(t, pk, side="right")
    return rise, tail


def pulse_check(suffix="own4"):
    """pulse-check: how many pulses the frozen setting places under each isolated SCR (measure():
    no other peak >= 0.02 uS from 5 s before its onset to its half-recovery), on the 50 real
    participants and on the 50 test sessions, where the true pulses are known too.

    An isolated SCR is one response, so the physiological expectation is about one pulse on its rise
    and none on its recovery.  The test sessions show what the solver does there when the truth is
    known (true pulses below a counted peak can still sit inside, as on real data).  More pulses per
    isolated SCR on real data than on the test sessions point to false pulses that are particular to
    real signals; the same number points to the excess coming from elsewhere (overlapping responses
    split apart where counting merges them).  Counted for all returned pulses (>= 0.02 uS) and for
    those >= BIG_uS.  Saved: data/pulse_check_isolated.csv (one row per isolated SCR)."""
    from . import runs
    from .tbme_score import event_heights
    rows = []
    real = pd.read_csv(REAL_CSV)
    pulses = pd.read_csv(runs.PULSES_CSV)
    for pid, m in real.groupby("source_id"):
        p = pulses[pulses.id == pid]
        r, t = _in_windows(m, p.time_s)
        rb, tb = _in_windows(m, p.time_s, p.height_uS, BIG_uS)
        rows.append(pd.DataFrame({"data": "real", "id": pid, "shape": "", "amp_uS": m.amp_uS.to_numpy(),
                                  "half_s": m.half_s.to_numpy(), "est_rise": r, "est_tail": t,
                                  "est_rise_big": rb, "est_tail_big": tb}))
    root = runs.FINAL_OUT / runs.FINAL6_SETTING
    for pid, g, m in _test_sessions(gen.selected_ids(), suffix):
        if not len(m):
            continue
        name = f"V{pid}_{suffix}"
        ev = pd.read_csv(root / f"{name}_events.csv")
        win = pd.read_csv(root / f"{name}_windows.csv")
        h = event_heights(ev, win)
        tr = np.load(ts.TRUTH_DIR / f"{name}.npz")
        r, t = _in_windows(m, ev.time_s)
        rb, tb = _in_windows(m, ev.time_s, h, BIG_uS)
        qr, qt = _in_windows(m, tr["ev_t"], tr["ev_a"], cfg.MIN_EVENT_uS)
        qrb, qtb = _in_windows(m, tr["ev_t"], tr["ev_a"], BIG_uS)
        rows.append(pd.DataFrame({"data": "test", "id": pid, "shape": g, "amp_uS": m.amp_uS.to_numpy(),
                                  "half_s": m.half_s.to_numpy(), "est_rise": r, "est_tail": t,
                                  "est_rise_big": rb, "est_tail_big": tb, "true_rise": qr, "true_tail": qt,
                                  "true_rise_big": qrb, "true_tail_big": qtb}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(PULSE_CHECK_CSV, index=False)

    def dist(v):
        v = np.asarray(v)
        return (f"0: {(v == 0).mean():.0%}, 1: {(v == 1).mean():.0%}, 2: {(v == 2).mean():.0%},"
                f" 3+: {(v >= 3).mean():.0%}; mean {v.mean():.2f}")

    print(f"\npulses under each isolated SCR (rise: {PRE_S:g} s before onset to peak; recovery: peak to"
          f" half-recovery); frozen setting {runs.FINAL6_SETTING}")
    for data, d in out.groupby("data", sort=False):
        per = d.groupby("id")
        print(f"\n{data}: {len(d)} isolated SCRs from {d.id.nunique()} {'participants' if data == 'real' else 'test sessions'}"
              f" (amplitude median {d.amp_uS.median():.3f} uS)")
        print(f"  returned, rise:            {dist(d.est_rise)}")
        print(f"  returned >= {BIG_uS} uS, rise:   {dist(d.est_rise_big)}")
        print(f"  returned, recovery:        {dist(d.est_tail)}")
        print(f"  returned >= {BIG_uS} uS, recovery: {dist(d.est_tail_big)}")
        if data == "test":
            print(f"  true >= {cfg.MIN_EVENT_uS} uS, rise:      {dist(d.true_rise)}")
            print(f"  true >= {cfg.MIN_EVENT_uS} uS, recovery:  {dist(d.true_tail)}")
            print(f"  true >= {BIG_uS} uS, rise:      {dist(d.true_rise_big)}")
        print(f"  per {'participant' if data == 'real' else 'session'}, median of the mean returned on the rise:"
              f" {per.est_rise.mean().median():.2f}, on the recovery: {per.est_tail.mean().median():.2f}")
    print("saved", PULSE_CHECK_CSV.name)
    return out
