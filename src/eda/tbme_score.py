"""
Scoring the Amin & Faghih (2021) code on the synthetic sessions (tbme_synth.py).

For each session S<k> the events the published code returned (cluster output, S<k>_events.csv
and S<k>_windows.csv) are compared with the impulses that were put in (data/tbme_synth/S<k>.npz):

  matching     one-to-one, closest first, within 1 s (match below, the same rule used for
               every earlier test)
  recall       share of true impulses recovered; also over impulses >= 0.02 uS only, the
               smallest response peak detection could ever see
  precision    share of returned events that are real
  timing       mean |time error| of matched pairs
  amplitude    Spearman rank correlation of matched pairs.  Ranks, because the published code's
               u is the input to its ODE and our a_i is a response height in uS: the units
               differ, the ordering should not
  tau          median over windows of the fitted tau_rise / tau_decay vs the truth
  fit          R2 of the reconstruction, from the windows table

Pass rule, fixed before the run: median F1 >= 0.7 in the "prior" and "moderate" tau groups.
The "far" group shows what happens when the priors point the wrong way; it is reported, not
required to pass.

Saved: data/tbme_synth_scores.csv (one row per session, with its settings and source).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as cfg
from . import tbme_real
from . import tbme_synth as ts

OUTPUT = cfg.ROOT / "hpc" / "synth" / "output"
SCORES_CSV = cfg.ROOT / "data" / "tbme_synth_scores.csv"
F1_PASS = 0.7


def match(true_t, est_t, tol=None):
    """One-to-one matching of true and recovered impulse times -> (true idx, est idx).

    All pairs closer than tol are ranked by distance and taken greedily, so each impulse is
    used at most once and the closest pairs win.
    """
    tol = cfg.MERGE_WINDOW_S if tol is None else tol
    i, j = np.nonzero(np.abs(true_t[:, None] - est_t[None, :]) <= tol)
    order = np.argsort(np.abs(true_t[i] - est_t[j]))
    used_t, used_e, pairs = set(), set(), []
    for a, b in zip(i[order], j[order]):
        if a not in used_t and b not in used_e:
            used_t.add(a), used_e.add(b), pairs.append((a, b))
    pairs.sort()
    return (np.array([p[0] for p in pairs], dtype=int),
            np.array([p[1] for p in pairs], dtype=int))


def truth_file(name):
    """S13_m2 -> truth of S13; the _pp sessions have their own (capped, filtered) truth."""
    base = name.split("_")[0]
    return ts.TRUTH_DIR / f"{base}_pp.npz" if name.endswith("_pp") else ts.TRUTH_DIR / f"{base}.npz"


def score_session(name, out_dir=OUTPUT):
    """One synthetic session -> dict of metrics (or None if its cluster output is missing)."""
    ev_f, win_f = out_dir / f"{name}_events.csv", out_dir / f"{name}_windows.csv"
    if not ev_f.exists():
        return None
    tr = np.load(truth_file(name))
    ev = pd.read_csv(ev_f)
    win = pd.read_csv(win_f)
    true_t, true_a = tr["ev_t"], tr["ev_a"]
    est_t, est_a = ev.time_s.to_numpy(), ev.amplitude.to_numpy()

    ti, ei = match(true_t, est_t)
    n_true, n_est, n_hit = len(true_t), len(est_t), len(ti)
    recall, precision = n_hit / max(n_true, 1), n_hit / max(n_est, 1)
    visible = true_a >= cfg.MIN_EVENT_uS
    return {
        "n_true": n_true, "n_est": n_est, "n_hit": n_hit,
        "recall": recall, "precision": precision,
        "f1": 2 * recall * precision / max(recall + precision, 1e-12),
        "recall_visible": np.isin(np.flatnonzero(visible), ti).mean() if visible.any() else np.nan,
        "timing_err_s": np.abs(true_t[ti] - est_t[ei]).mean() if n_hit else np.nan,
        "amp_spearman": spearmanr(true_a[ti], est_a[ei])[0] if n_hit > 2 else np.nan,
        "tau_rise_fit": win.tau_rise.median(), "tau_decay_fit": win.tau_decay.median(),
        "tau_rise_err_pct": 100 * (win.tau_rise.median() / float(tr["tau_rise"]) - 1),
        "tau_decay_err_pct": 100 * (win.tau_decay.median() / float(tr["tau_decay"]) - 1),
        "tau_rise_spread": win.tau_rise.max() - win.tau_rise.min(),
        "tau_decay_spread": win.tau_decay.max() - win.tau_decay.min(),
        "r2_min": win.r2.min(), "r2_median": win.r2.median(),
        "windows_authors_rule": (win.selection == "authors").mean(),
    }


def run(out_dir=OUTPUT):
    """Score every session of the synthetic cohort that has cluster output; save and summarise."""
    cohort = pd.read_csv(ts.COHORT_CSV)
    rows, missing = [], []
    for r in cohort.itertuples():
        s = score_session(r.session, out_dir)
        if s is None:
            missing.append(r.session)
            continue
        rows.append({"session": r.session, "source_id": r.source_id, "group": r.group,
                     "tau_group": r.tau_group, "tau_rise": r.tau_rise, "tau_decay": r.tau_decay,
                     "source_level_uS": r.real_level_uS, "source_scr_pm": r.real_scr_pm,
                     "source_amp_med_uS": r.real_scr_amp_med_uS, **s})
    if missing:
        print(f"no output yet for {len(missing)}: {' '.join(missing)}")
    if not rows:
        return None
    sc = pd.DataFrame(rows)
    sc.to_csv(SCORES_CSV, index=False)

    cols = ["recall", "recall_visible", "precision", "f1", "timing_err_s", "amp_spearman",
            "tau_rise_fit", "tau_decay_fit", "tau_rise_err_pct", "tau_decay_err_pct", "r2_min"]
    g = sc.groupby("tau_group")
    print(f"{len(sc)} sessions scored, median per tau group:")
    print(pd.concat([g.size().rename("n"), g[["tau_rise", "tau_decay"]].median(),
                     g[cols].median()], axis=1).T.to_string(float_format=lambda v: f"{v:.3g}"))
    print("worst 5 sessions by F1:")
    print(sc.nsmallest(5, "f1")[["session", "source_id", "tau_group", "source_level_uS",
                                 "source_scr_pm", "recall", "precision", "f1", "r2_min"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    neg = sc[sc.source_level_uS < 0]
    if len(neg):
        print(f"sessions copied from negative-offset participants: median F1 {neg.f1.median():.3g} "
              f"(n={len(neg)}) vs {sc[sc.source_level_uS >= 0].f1.median():.3g} for the rest")
    ok = {t: g.f1.median().get(t, np.nan) >= F1_PASS for t in ("prior", "moderate")}
    print(f"pass rule (median F1 >= {F1_PASS} in prior and moderate): "
          f"{'PASS' if all(ok.values()) else 'FAIL'} {ok}")
    print("saved", SCORES_CSV.name)
    return sc


SHIFT_CSV = cfg.ROOT / "data" / "tbme_synth_shift_scores.csv"


def shift_test(out_dir=OUTPUT):
    """The low-level sessions as run (none), offset removed (m0) and lifted to 2 uS (m2)."""
    cohort = pd.read_csv(ts.COHORT_CSV)
    low = cohort[cohort.real_level_uS < ts.LOW_LEVEL_uS]
    rows = []
    for r in low.itertuples():
        for suf in ("", *(f"_{k}" for k in ts.SHIFT_TARGETS)):
            s = score_session(r.session + suf, out_dir)
            if s is None:
                print(f"no output yet for {r.session + suf}")
                continue
            rows.append({"session": r.session, "variant": suf.lstrip("_") or "none",
                         "source_id": r.source_id, "source_level_uS": r.real_level_uS,
                         "tau_group": r.tau_group, **s})
    if not rows:
        return None
    sc = pd.DataFrame(rows)
    sc.to_csv(SHIFT_CSV, index=False)
    cols = ["f1", "recall", "precision", "tau_rise_spread", "tau_decay_spread", "r2_min"]
    print("per session:")
    print(sc.pivot(index="session", columns="variant", values=["f1", "tau_decay_spread", "r2_min"])
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print("median over the low-level sessions:")
    print(sc.groupby("variant")[cols].median().T.to_string(float_format=lambda v: f"{v:.3g}"))
    ref = pd.read_csv(SCORES_CSV)
    ref = ref[ref.source_level_uS >= ts.LOW_LEVEL_uS]
    print(f"reference, the other {len(ref)} sessions (unshifted): " + ", ".join(
        f"{c} {ref[c].median():.3g}" for c in cols if c in ref))
    print("saved", SHIFT_CSV.name)
    return sc


# ------------------------------------------------------------------ false events
FALSE_CSV = cfg.ROOT / "data" / "tbme_synth_false_events.csv"
CUTOFF_CSV = cfg.ROOT / "data" / "tbme_synth_cutoff.csv"
NEAR_S = 2.0                                      # "close to a true impulse"
MERGE_S = (0.5, 1.0, 1.5, 2.0)                    # merge windows tried


def preprocessed_name(session, out_dir=OUTPUT):
    """The run that matches the chosen preprocessing: offset-removed copy if there is one."""
    return f"{session}_m0" if (out_dir / f"{session}_m0_events.csv").exists() else session


def _f1(true_t, est_t):
    ti, _ = match(true_t, est_t)
    r, p = len(ti) / max(len(true_t), 1), len(ti) / max(len(est_t), 1)
    return 2 * r * p / max(r + p, 1e-12), r, p


def merge_close(t, a, gap_s):
    """Events closer than gap_s joined into one: amplitudes summed, time of the largest."""
    if not len(t):
        return t, a
    groups = np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > gap_s) + 1)
    return (np.array([t[g][np.argmax(a[g])] for g in groups]),
            np.array([a[g].sum() for g in groups]))


def false_events(out_dir=OUTPUT):
    """Every returned event labelled hit / false; false ones classified by what lies near them.

    duplicate   a true impulse within NEAR_S s, already matched to another event
    off_time    a true impulse within NEAR_S s, unmatched (found, but more than 1 s off)
    isolated    no true impulse within NEAR_S s
    """
    cohort = pd.read_csv(ts.COHORT_CSV)
    rows = []
    for r in cohort.itertuples():
        name = preprocessed_name(r.session, out_dir)
        tr = np.load(ts.TRUTH_DIR / f"{r.session}.npz")
        ev = pd.read_csv(out_dir / f"{name}_events.csv")
        true_t, est_t, est_a = tr["ev_t"], ev.time_s.to_numpy(), ev.amplitude.to_numpy()
        ti, ei = match(true_t, est_t)
        hit = np.isin(np.arange(len(est_t)), ei)
        d = np.abs(est_t[:, None] - true_t[None, :])
        near = d.argmin(1)
        kind = np.where(hit, "hit", np.where(d.min(1) > NEAR_S, "isolated",
                        np.where(np.isin(near, ti), "duplicate", "off_time")))
        rows.append(pd.DataFrame({"session": r.session, "run": name, "tau_group": r.tau_group,
                                  "time_s": est_t, "amplitude": est_a,
                                  "amp_rank": pd.Series(est_a).rank(pct=True).to_numpy(),
                                  "dist_true_s": d.min(1), "kind": kind}))
    fe = pd.concat(rows, ignore_index=True)
    fe.to_csv(FALSE_CSV, index=False)
    false = fe[fe.kind != "hit"]
    print(f"{len(fe)} returned events, {len(false)} false ({len(false) / len(fe):.0%})")
    print("false events by kind:", false.kind.value_counts().to_dict())
    print("amplitude, median [10-90%]:")
    for k, g in fe.groupby(fe.kind == "hit"):
        a = g.amplitude
        print(f"  {'hits ' if k else 'false'}  {a.median():.3g} [{a.quantile(.1):.3g}-"
              f"{a.quantile(.9):.3g}], within-session rank median {g.amp_rank.median():.2f}")
    print("share of false events among the smallest x% of each session's events:")
    for q in (0.1, 0.25, 0.5):
        s = fe[fe.amp_rank <= q]
        print(f"  smallest {q:.0%}: {(s.kind != 'hit').mean():.0%} false")
    print("saved", FALSE_CSV.name)
    return fe


def cutoff_test(out_dir=OUTPUT):
    """Minimum-size cut-offs and merging of close events, chosen on half, checked on the other.

    Sessions are split in two halves balanced by tau group (alternating within each group).
    For each half the cut-off with the best median F1 is chosen, then scored on the other half.
    """
    fe = pd.read_csv(FALSE_CSV) if FALSE_CSV.exists() else false_events(out_dir)
    cohort = pd.read_csv(ts.COHORT_CSV)
    half = {s: k % 2 for _, g in cohort.groupby("tau_group")
            for k, s in enumerate(g.session)}
    truth = {s: np.load(ts.TRUTH_DIR / f"{s}.npz")["ev_t"] for s in cohort.session}
    cuts = np.unique(np.round(np.quantile(fe.amplitude, np.linspace(0, 0.5, 21)), 6))

    rows = []
    for s, g in fe.groupby("session"):
        t, a = g.time_s.to_numpy(), g.amplitude.to_numpy()
        for c in cuts:
            f, r, p = _f1(truth[s], t[a >= c])
            rows.append({"session": s, "half": half[s], "rule": "cut", "value": c,
                         "f1": f, "recall": r, "precision": p})
        for m in MERGE_S:
            tm, am = merge_close(t, a, m)
            f, r, p = _f1(truth[s], tm)
            rows.append({"session": s, "half": half[s], "rule": "merge", "value": m,
                         "f1": f, "recall": r, "precision": p})
    res = pd.DataFrame(rows)
    res.to_csv(CUTOFF_CSV, index=False)

    base = res[(res.rule == "cut") & (res.value == cuts[0])]
    print(f"no cut-off (all events): median F1 {base.f1.median():.3f}, "
          f"recall {base.recall.median():.3f}, precision {base.precision.median():.3f}")
    med = res.groupby(["rule", "value", "half"])[["f1", "recall", "precision"]].median()
    for h in (0, 1):
        cut = med.xs(("cut", h), level=("rule", "half")).f1.idxmax()
        other = med.loc[("cut", cut, 1 - h)]
        own0 = med.loc[("cut", cuts[0], 1 - h)]
        print(f"cut-off chosen on half {h}: {cut:.4g} -> on half {1 - h}: F1 {own0.f1:.3f} -> "
              f"{other.f1:.3f}, recall {own0.recall:.3f} -> {other.recall:.3f}, "
              f"precision {own0.precision:.3f} -> {other.precision:.3f}")
    print("merging close events (both halves):")
    mg = res[res.rule == "merge"].groupby("value")[["f1", "recall", "precision"]].median()
    print(mg.to_string(float_format=lambda v: f"{v:.3f}"))
    print("saved", CUTOFF_CSV.name)
    return res


# ------------------------------------------------------------------ window length and tau
WINDOW_CSV = cfg.ROOT / "data" / "tbme_synth_window_scores.csv"
WINDOW_SESSIONS = ("S9", "S15", "S24", "S33")    # far group, positive level, rates 4-10/min
WINDOW_LENGTHS = (200, 400, 800)


def window_test(out_dir=OUTPUT):
    """Does a longer window free tau from the priors? Same sessions at 200 / 400 / 800 s."""
    rows = []
    for s in WINDOW_SESSIONS:
        tr = np.load(ts.TRUTH_DIR / f"{s}.npz")
        for w in WINDOW_LENGTHS:
            name = s if w == 200 else f"{s}_w{w}"
            sc = score_session(name, out_dir)
            if sc is None:
                print(f"no output yet for {name}")
                continue
            secs = pd.read_csv(out_dir / f"{name}_windows.csv").seconds.sum()
            rows.append({"session": s, "window_s": w, "tau_rise_true": float(tr["tau_rise"]),
                         "tau_decay_true": float(tr["tau_decay"]), "solver_s": secs, **sc})
    if not rows:
        return None
    wt = pd.DataFrame(rows)
    wt.to_csv(WINDOW_CSV, index=False)
    print(wt[["session", "window_s", "tau_rise_true", "tau_rise_fit", "tau_decay_true",
              "tau_decay_fit", "f1", "recall", "precision", "r2_min", "solver_s"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("median by window length:")
    print(wt.groupby("window_s")[["tau_rise_err_pct", "tau_decay_err_pct", "f1", "precision",
                                  "solver_s"]].median().to_string(float_format=lambda v: f"{v:.3g}"))
    print("saved", WINDOW_CSV.name)
    return wt



# ------------------------------------------------------------------ sampling rate and tau
RATE_CSV = cfg.ROOT / "data" / "tbme_synth_rate_scores.csv"


def rate_test(out_dir=OUTPUT, fsy=ts.RATE_FSY):
    """Same sessions at 2 Hz (as run) and at fsy: does a finer rate let tau leave the priors?"""
    cohort = pd.read_csv(ts.COHORT_CSV).set_index("session")
    # S9_f4 (100 s windows) or S9_f4_w200 (200 s windows, the first 4 Hz run: the job script
    # passed 200 s explicitly)
    runs = {p.name[:-len("_events.csv")] for p in out_dir.glob(f"S*_f{fsy}*_events.csv")}
    names = sorted({r.split("_")[0] for r in runs}, key=lambda n: int(n[1:]))
    rows = []
    for s in names:
        for hz, name in ((2, preprocessed_name(s, out_dir)),
                         *((fsy, r) for r in sorted(runs) if r.split("_")[0] == s)):
            sc = score_session(name, out_dir)
            secs = pd.read_csv(out_dir / f"{name}_windows.csv").seconds.sum()
            rows.append({"session": s, "fs_hz": hz, "run": name, "tau_group": cohort.loc[s, "tau_group"],
                         "tau_rise_true": cohort.loc[s, "tau_rise"],
                         "tau_decay_true": cohort.loc[s, "tau_decay"], "solver_s": secs, **sc})
    if not rows:
        print("no rate-test output found")
        return None
    rt = pd.DataFrame(rows)
    rt.to_csv(RATE_CSV, index=False)
    print(rt[["session", "tau_group", "run", "tau_rise_true", "tau_rise_fit", "tau_decay_true",
              "tau_decay_fit", "tau_decay_spread", "f1", "precision", "recall", "solver_s"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("median by tau group and rate:")
    print(rt.groupby(["tau_group", "fs_hz"])[["tau_rise_err_pct", "tau_decay_err_pct", "f1",
                                              "precision", "recall", "solver_s"]].median()
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print("saved", RATE_CSV.name)
    return rt


# ------------------------------------------------------------------ tail fingerprint
TAIL_CSV = cfg.ROOT / "data" / "tbme_tail_fingerprint.csv"
REAL_OUTPUT = cfg.ROOT / "hpc" / "results"
TAIL_WIN_S = (1.0, 6.0)          # after the published minimum peak distance, up to ~2 x 2.77 s
TAIL_RATIOS = (1.5, 2.0, 3.0)    # 2.0 is the main one, fixed before looking; others = sensitivity
PRIOR_DECAY = ts.PRIOR["decay"][0]


def tail_fraction(t, a, ratio, win=TAIL_WIN_S):
    """Share of events lying win s after an event at least `ratio` times larger.

    When the true decay is longer than the model's, the model has to prop up each long tail
    with extra, smaller pulses shortly after the big one; this counts them.
    """
    if len(t) < 2:
        return np.nan
    dt = t[None, :] - t[:, None]                   # [i, j] = t_j - t_i, j after i
    big = a[:, None] >= ratio * a[None, :]
    on_tail = ((dt >= win[0]) & (dt <= win[1]) & big).any(0)
    return on_tail.mean()


def tail_test(out_dir=OUTPUT, real_dir=REAL_OUTPUT):
    """Tail fingerprint on synthetic sessions (known tau) and on real participants run so far.

    If the fingerprint rises with the true decay in the synthetic sessions, its value on a
    real participant says whether their decay is near the prior (2.77 s) or longer.
    """
    cohort = pd.read_csv(ts.COHORT_CSV)
    rows = []
    for r in cohort.itertuples():
        name = preprocessed_name(r.session, out_dir)
        ev = pd.read_csv(out_dir / f"{name}_events.csv")
        decay_set = ("far" if r.tau_group == "far" else "prior" if r.tau_group == "prior"
                     else "moderate_short" if r.tau_decay < PRIOR_DECAY else "moderate_long")
        rows.append({"name": r.session, "kind": "synthetic", "decay_set": decay_set,
                     "tau_decay_true": r.tau_decay, "n_events": len(ev),
                     **{f"tail_{k}": tail_fraction(ev.time_s.to_numpy(), ev.amplitude.to_numpy(), k)
                        for k in TAIL_RATIOS}})
    for f in sorted(real_dir.glob("P*_events.csv")):
        ev = pd.read_csv(f)
        rows.append({"name": f.name.split("_")[0], "kind": "real", "decay_set": "real",
                     "tau_decay_true": np.nan, "n_events": len(ev),
                     **{f"tail_{k}": tail_fraction(ev.time_s.to_numpy(), ev.amplitude.to_numpy(), k)
                        for k in TAIL_RATIOS}})
    tf = pd.DataFrame(rows)
    tf.to_csv(TAIL_CSV, index=False)

    syn = tf[tf.kind == "synthetic"]
    cols = [f"tail_{k}" for k in TAIL_RATIOS]
    order = ["moderate_short", "prior", "moderate_long", "far"]
    print("synthetic, tail fraction by true decay (median [min-max]):")
    for d in order:
        g = syn[syn.decay_set == d]
        print(f"  {d:15s} n={len(g):2d} decay {g.tau_decay_true.median():.2f} s   " + "   ".join(
            f"x{k}: {g[c].median():.3f} [{g[c].min():.3f}-{g[c].max():.3f}]"
            for k, c in zip(TAIL_RATIOS, cols)))
    print("rank correlation with the true decay (does the fingerprint track it?): " + ", ".join(
        f"x{k} {spearmanr(syn.tau_decay_true, syn[c])[0]:.2f}" for k, c in zip(TAIL_RATIOS, cols)))
    real = tf[tf.kind == "real"]
    if len(real):
        print("real participants:")
        for _, r in real.iterrows():
            vals = [r[c] for c in cols]
            pr = [100 * (syn.loc[syn.decay_set == "prior", c] <= v).mean() for c, v in zip(cols, vals)]
            fr = [100 * (syn.loc[syn.decay_set == "far", c] <= v).mean() for c, v in zip(cols, vals)]
            print(f"  {r['name']}: " + "   ".join(
                f"x{k}: {v:.3f} (above {p:.0f}% of prior, {q:.0f}% of far sessions)"
                for k, v, p, q in zip(TAIL_RATIOS, vals, pr, fr)))
    print("saved", TAIL_CSV.name)
    return tf


# ------------------------------------------------------------------ does each pulse matter?
IMPORTANCE_CSV = cfg.ROOT / "data" / "tbme_synth_importance.csv"
IMP_RULES_CSV = cfg.ROOT / "data" / "tbme_synth_importance_rules.csv"
SEG_S = (5.0, 15.0)              # stretch scored around a pulse: 5 s before to 15 s after
NEIGH_S = 15.0                   # pulses this close may refit to absorb it


def bateman(t, tau_rise, tau_decay):
    """The model's response shape (unscaled); zero before the pulse."""
    return np.where(t >= 0, np.exp(-np.clip(t, 0, None) / tau_decay)
                    - np.exp(-np.clip(t, 0, None) / tau_rise), 0.0)


def pulse_importance(sig, ev, tau_rise, tau_decay):
    """Per pulse: fit loss when it is removed and its neighbours refit, in noise units.

    score_i = (SSE without i - SSE with i) / sigma^2, both fits non-negative least squares on
    the model's own phasic target (y - tonic) over [t_i - 5 s, t_i + 15 s], with the pulses
    within 15 s free and a free constant.  sigma = SD of the session's reconstruction error.
    A pulse the neighbours can replace scores near 0; a real response should not be
    replaceable.  Also returns how well the plain Bateman shape at the session's tau rebuilds
    the model's phasic output (a check that we score with the model's own kernel).
    """
    from scipy.optimize import nnls
    ty = sig.time_s.to_numpy()
    target = (sig.y - sig.tonic).to_numpy()
    et, ea = ev.time_s.to_numpy(), ev.amplitude.to_numpy()
    sigma2 = np.var((sig.y - sig.y_rec).to_numpy())
    X = bateman(ty[:, None] - et[None, :], tau_rise, tau_decay)
    pred = X @ ea
    k = pred @ sig.phasic.to_numpy() / max(pred @ pred, 1e-12)
    ph = sig.phasic.to_numpy()
    kernel_r2 = 1 - np.sum((ph - k * pred) ** 2) / np.sum((ph - ph.mean()) ** 2)
    score = np.empty(len(et))
    for i in range(len(et)):
        seg = (ty >= et[i] - SEG_S[0]) & (ty <= et[i] + SEG_S[1])
        nb = np.flatnonzero(np.abs(et - et[i]) <= NEIGH_S)
        A = np.column_stack([X[seg][:, nb], np.ones(seg.sum()), -np.ones(seg.sum())])
        b = target[seg]
        with_i = nnls(A, b)[1] ** 2
        without_i = nnls(np.delete(A, np.flatnonzero(nb == i)[0], axis=1), b)[1] ** 2
        score[i] = (without_i - with_i) / max(sigma2, 1e-12)
    return score, kernel_r2


def auc(pos, neg):
    """Probability that a random true pulse outscores a random false one (0.5 = useless)."""
    from scipy.stats import mannwhitneyu
    if not len(pos) or not len(neg):
        return np.nan
    return mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def importance_test(out_dir=OUTPUT):
    """Size cut-off vs 'does the pulse matter', both chosen on half the sessions, scored on the other."""
    cohort = pd.read_csv(ts.COHORT_CSV)
    half = {s: k % 2 for _, g in cohort.groupby("tau_group") for k, s in enumerate(g.session)}
    rows, checks = [], []
    for r in cohort.itertuples():
        name = preprocessed_name(r.session, out_dir)
        sig = pd.read_csv(out_dir / f"{name}_signals.csv")
        ev = pd.read_csv(out_dir / f"{name}_events.csv")
        win = pd.read_csv(out_dir / f"{name}_windows.csv")
        tr_, td_ = win.tau_rise.median(), win.tau_decay.median()
        sc, kr2 = pulse_importance(sig, ev, tr_, td_)
        checks.append(kr2)
        true_t = np.load(ts.TRUTH_DIR / f"{r.session}.npz")["ev_t"]
        _, ei = match(true_t, ev.time_s.to_numpy())
        hit = np.isin(np.arange(len(ev)), ei)
        rows.append(pd.DataFrame({"session": r.session, "run": name, "half": half[r.session],
                                  "tau_group": r.tau_group, "time_s": ev.time_s,
                                  "amplitude": ev.amplitude, "importance": sc, "hit": hit}))
    imp = pd.concat(rows, ignore_index=True)
    imp.to_csv(IMPORTANCE_CSV, index=False)
    print(f"kernel check, Bateman at each session's tau rebuilds the model's phasic part: "
          f"R2 median {np.median(checks):.3f} (min {np.min(checks):.3f})")
    print("separating true from false pulses (AUC, 0.5 = useless, 1 = perfect):")
    for col in ("amplitude", "importance"):
        print(f"  {col:10s} all {auc(imp.loc[imp.hit, col], imp.loc[~imp.hit, col]):.3f}   " +
              "   ".join(f"{g} {auc(d.loc[d.hit, col], d.loc[~d.hit, col]):.3f}"
                         for g, d in imp.groupby("tau_group")))

    truth = {s: np.load(ts.TRUTH_DIR / f"{s}.npz")["ev_t"] for s in cohort.session}
    out = []
    for col in ("amplitude", "importance"):
        grid = np.unique(np.quantile(imp[col], np.linspace(0, 0.6, 31)))
        for c in grid:
            for s, g in imp.groupby("session"):
                f, rc, p = _f1(truth[s], g.time_s.to_numpy()[g[col].to_numpy() >= c])
                out.append({"rule": col, "value": c, "session": s, "half": half[s],
                            "f1": f, "recall": rc, "precision": p})
    res = pd.DataFrame(out)
    res.to_csv(IMP_RULES_CSV, index=False)
    med = res.groupby(["rule", "value", "half"])[["f1", "recall", "precision"]].median()
    for col in ("amplitude", "importance"):
        for h in (0, 1):
            m = med.xs((col, h), level=("rule", "half"))
            c = m.f1.idxmax()
            o = med.loc[(col, c, 1 - h)]
            b = med.loc[(col, m.index.min(), 1 - h)]
            print(f"{col:10s} chosen on half {h} ({c:.3g}) -> half {1 - h}: F1 {b.f1:.3f} -> "
                  f"{o.f1:.3f}, precision {b.precision:.3f} -> {o.precision:.3f}, "
                  f"recall {b.recall:.3f} -> {o.recall:.3f}")
    print("saved", IMPORTANCE_CSV.name, IMP_RULES_CSV.name)
    return imp


# ------------------------------------------------------------------ does the tonic stay slow?
TONIC_CSV = cfg.ROOT / "data" / "tbme_tonic_check.csv"
REAL_RUNS = cfg.ROOT / "hpc" / "real" / "output"
FAST_HZ = 0.05                   # faster than a 20 s cycle: response time scale, not tonic


def fast_share(x, fs=ts.tx.FSY):
    """Share of a curve's variance (detrended) at frequencies above FAST_HZ."""
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return p[f > FAST_HZ].sum() / max(p[1:].sum(), 1e-30)


def tonic_metrics(name, out_dir=OUTPUT):
    """Estimated vs true tonic of one synthetic run (see tonic_test)."""
    sig = pd.read_csv(out_dir / f"{name}_signals.csv")
    tr = np.load(truth_file(name))
    true_tonic = np.interp(sig.time_s, tr["t2"], tr["tonic2"])
    true_phasic = np.interp(sig.time_s, tr["t2"], tr["phasic2"])
    tonic = sig.tonic.to_numpy() - np.median(sig.y - true_tonic - true_phasic)
    err = tonic - true_tonic
    return {"tonic_rmse_uS": np.sqrt(np.mean(err ** 2)),
            "phasic_sd_uS": true_phasic.std(),
            "phasic_taken": np.cov(err, true_phasic)[0, 1] / max(true_phasic.var(), 1e-30),
            "fast_share_est": fast_share(tonic), "fast_share_true": fast_share(true_tonic),
            "fast_share_signal": fast_share(sig.y.to_numpy())}


def tonic_test(out_dir=OUTPUT, real_dir=REAL_RUNS):
    """Estimated tonic vs the true floor (synthetic) and its wobble (synthetic and real).

    phasic_taken: slope of (estimated - true tonic) on the true phasic, i.e. the share of each
    response the tonic absorbs (0 = clean split, 1 = the tonic takes the whole response).
    Shifted runs (_m0) are put back on the truth's scale by their median offset.
    """
    cohort = pd.read_csv(ts.COHORT_CSV)
    rows = [{"name": r.session, "kind": "synthetic", "tau_group": r.tau_group,
             **tonic_metrics(preprocessed_name(r.session, out_dir), out_dir)}
            for r in cohort.itertuples()]
    for f in sorted(real_dir.glob("P*_signals.csv")):
        sig = pd.read_csv(f)
        rows.append({"name": f.name.split("_")[0], "kind": "real", "tau_group": "real",
                     "fast_share_est": fast_share(sig.tonic.to_numpy()),
                     "fast_share_signal": fast_share(sig.y.to_numpy())})
    tc = pd.DataFrame(rows)
    tc.to_csv(TONIC_CSV, index=False)

    cols = ["tonic_rmse_uS", "phasic_sd_uS", "phasic_taken", "fast_share_est",
            "fast_share_true", "fast_share_signal"]
    print(f"tonic check (fast = above {FAST_HZ} Hz), synthetic median by tau group:")
    print(tc[tc.kind == "synthetic"].groupby("tau_group")[cols].median()
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("real participants:")
    print(tc[tc.kind == "real"][["name", "fast_share_est", "fast_share_signal"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("saved", TONIC_CSV.name)
    return tc


# ------------------------------------------------------------------ tau forced to the truth
FIXED_CSV = cfg.ROOT / "data" / "tbme_synth_fixed_tau.csv"
FIXED_SESSIONS = ("S4", "S16", "S25", "S34", "S9", "S15", "S24", "S33")   # 4 prior, 4 far


def tau_suffix(tau_rise, tau_decay):
    """Output suffix tbme_run.m gives a run with tau fixed there."""
    return f"_t{round(100 * tau_rise)}_{round(100 * tau_decay)}"


def fixed_tau_test(out_dir=OUTPUT):
    """Free tau (published bounds) vs tau forced to the truth, and far sessions forced to the prior.

    If forcing the true tau brings the far sessions up to the prior group (F1, little response
    absorbed by the tonic), the tau mismatch is the cause; if not, the problem is elsewhere.
    """
    cohort = pd.read_csv(ts.COHORT_CSV).set_index("session")
    prior = (ts.PRIOR["rise"][0], ts.PRIOR["decay"][0])
    rows = []
    for s in FIXED_SESSIONS:
        r = cohort.loc[s]
        runs = {"free": s, "true": s + tau_suffix(r.tau_rise, r.tau_decay)}
        if r.tau_group == "far":
            runs["prior"] = s + tau_suffix(*prior)
        for tau, name in runs.items():
            sc = score_session(name, out_dir)
            if sc is None:
                print(f"  {name}: no output, skipped")
                continue
            rows.append({"session": s, "tau_group": r.tau_group, "tau": tau, "name": name,
                         "tau_rise_true": r.tau_rise, "tau_decay_true": r.tau_decay,
                         **sc, **tonic_metrics(name, out_dir)})
    ft = pd.DataFrame(rows)
    ft.to_csv(FIXED_CSV, index=False)
    cols = ["f1", "recall", "precision", "amp_spearman", "phasic_taken", "tonic_rmse_uS",
            "tau_rise_fit", "tau_decay_fit", "r2_median", "windows_authors_rule"]
    print("median by tau group and tau setting (free = published bounds):")
    print(ft.groupby(["tau_group", "tau"])[cols].median().to_string(float_format=lambda v: f"{v:.3f}"))
    print("saved", FIXED_CSV.name)
    return ft


# ------------------------------------------------------------------ the paper's preprocessing
PAPER_CSV = cfg.ROOT / "data" / "tbme_paper_test.csv"
PAPER_REAL_CSV = cfg.ROOT / "data" / "tbme_paper_real.csv"
REAL_PP_RUNS = cfg.ROOT / "hpc" / "real" / "output_pp"
PAPER_REAL_IDS = (53, 115, 26, 3, 80)


def paper_test(out_dir=OUTPUT):
    """Our earlier inputs vs the paper's preprocessing: the 8 test sessions and 5 real participants.

    Tau free in both.  The _pp sessions also have their largest responses capped, so a change
    in the synthetic scores is the two together.  f1_cut re-applies the size cut-off, which was
    chosen on the earlier inputs and may not carry over.
    """
    cohort = pd.read_csv(ts.COHORT_CSV).set_index("session")
    rows = []
    for s in ts.PAPER_SESSIONS:
        for inp, name in (("before", s), ("paper", f"{s}_pp")):
            sc = score_session(name, out_dir)
            if sc is None:
                print(f"  {name}: no output, skipped")
                continue
            ev = pd.read_csv(out_dir / f"{name}_events.csv")
            true_t = np.load(truth_file(name))["ev_t"]
            keep = ev.amplitude.to_numpy() >= cfg.TBME_CUT_U
            rows.append({"session": s, "tau_group": cohort.loc[s, "tau_group"], "input": inp,
                         "name": name, **sc, **tonic_metrics(name, out_dir),
                         "f1_cut": _f1(true_t, ev.time_s.to_numpy()[keep])[0]})
    pt = pd.DataFrame(rows)
    pt.to_csv(PAPER_CSV, index=False)
    cols = ["f1", "f1_cut", "recall", "precision", "amp_spearman", "phasic_taken", "tonic_rmse_uS",
            "tau_rise_fit", "tau_decay_fit", "r2_median", "windows_authors_rule"]
    print("synthetic, median by tau group and input:")
    print(pt.groupby(["tau_group", "input"])[cols].median().to_string(float_format=lambda v: f"{v:.3f}"))

    real = []
    for pid in PAPER_REAL_IDS:
        for inp, folder, suffix in (("before", REAL_RUNS, ""), ("paper", REAL_PP_RUNS, "_pp")):
            f = folder / f"P{pid}{suffix}_signals.csv"
            if not f.exists():
                print(f"  P{pid}{suffix}: no output, skipped")
                continue
            sig = pd.read_csv(f)
            tonic, signal = fast_share(sig.tonic.to_numpy()), fast_share(sig.y.to_numpy())
            real.append({"input": inp, "fast_share_tonic": tonic, "fast_share_signal": signal,
                         "tonic_over_signal": tonic / max(signal, 1e-30),
                         **tbme_real.summarise(pid, folder, suffix=suffix)})
    pr = pd.DataFrame(real)
    pr.to_csv(PAPER_REAL_CSV, index=False)
    print("real participants:")
    print(pr.set_index(["id", "input"])[["r2", "r2_worst_window", "tau_rise_median", "tau_decay_median",
                                         "windows_authors_rule", "events_pm_raw", "events_pm_kept",
                                         "counted_scr_pm", "tonic_over_signal", "solver_s"]]
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print("saved", PAPER_CSV.name, PAPER_REAL_CSV.name)
    return pt, pr
