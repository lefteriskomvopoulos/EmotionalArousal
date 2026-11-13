"""
Scoring the tuning rounds, every setting's cluster output on its 18 tuning sessions against the
known truth, and the diagnostics the rounds led to.

  score         own-score: rounds 1-5 on the first 18 (S, synth_v1.split), round 6 on the 18 built
                from the 50 (U, generator.export_tune50) -> data/own_tune_scores.csv
  scale         own-scale: each input's typical response size and setting C's gain (round 6)
  decay_check   own-decay-check: which sessions get too-long decays (rounds 4-5)
  error_check   own-errors: every miss and false pulse of one setting, by size and closeness

Cluster output: hpc/own/tune/output/<setting>/.  The settings of each round:
src/own/make_own_bundle.sh (comments).
"""
import numpy as np
import pandas as pd

from . import config as cfg
from . import tbme_export as tx
from . import tbme_synth as ts
from .generator import TUNE50_CSV
from .preprocess import input_scale
from .synth_v1 import split
from .tbme_score import event_heights

TUNE_CSV = cfg.ROOT / "data" / "own_tune_scores.csv"
SCALE_CSV = cfg.ROOT / "data" / "own_scale.csv"
DECAY_CSV = cfg.ROOT / "data" / "own_decay_check.csv"
DECAY_SESSIONS_CSV = cfg.ROOT / "data" / "own_decay_sessions.csv"
DECAY_SETTING = "own4_p1000_k14_f3"
ERRORS_CSV = cfg.ROOT / "data" / "own_errors.csv"
ERRORS_SUMMARY_CSV = cfg.ROOT / "data" / "own_errors_summary.csv"
CLOSE_S = 2.0      # a neighbour this near counts as close: twice the 1 s pruning / matching window
LARGE_uS = 0.05    # small = 0.02-0.05 uS (up to 5 resolution steps), large above


def score(root=cfg.ROOT / "hpc" / "own" / "tune" / "output"):
    """Each setting's cluster output on its 18 tuning sessions, against the known truth.
    Rounds 1-5 ran on the first 18 (S, own_split.csv), round 6 on the 18 built from the 50
    selected participants (U, tune50-export); each folder is scored on the set it holds, and the
    two sets are printed apart, since their medians are over different sessions."""
    from . import tbme_score as sc
    roles = split()
    sets = {"S": ("first 18 (S)", roles[roles.role == "tune"][["session", "tau_group"]])}
    if TUNE50_CSV.exists():
        sets["U"] = ("from the 50 (U)", pd.read_csv(TUNE50_CSV)[["session", "tau_group"]])
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        first = next(folder.glob("*_events.csv"), None)
        if first is None:
            print(f"  {folder.name}: empty, skipped")
            continue
        suffix = "_" + first.name.split("_")[1]                  # S11_own4_events.csv -> _own4
        label, sessions = sets[first.name[0]]
        for r in sessions.itertuples():
            name = f"{r.session}{suffix}"
            res = sc.score_session(name, folder)
            if res is None:
                print(f"  {folder.name}/{name}: no output, skipped")
                continue
            rows.append({"setting": folder.name, "tuning_set": label, "input": suffix[1:],
                         "session": r.session, "tau_group": r.tau_group, **res,
                         **sc.tonic_metrics(name, folder)})
    out = pd.DataFrame(rows)
    out.to_csv(TUNE_CSV, index=False)
    cols = ["f1_visible", "recall_visible", "f1", "recall", "precision", "timing_err_s",
            "amp_spearman", "tau_rise_err_pct",
            "tau_decay_err_pct", "phasic_taken", "windows_authors_rule", "lambda2_at_cap"]
    f = lambda v: f"{v:.3f}"
    for label, part in out.groupby("tuning_set", sort=False):
        print(f"\n=== tuning sessions {label} ===")
        print(part.groupby(["tau_group", "setting"])[cols].median().to_string(float_format=f))
        print("all 18, median:")
        print(part.groupby("setting")[cols].median().to_string(float_format=f))
    print("saved", TUNE_CSV.name)
    return out


def scale(suffix="own4"):
    """own-scale: each input's typical response size and the gain of setting C (round 6).

    The solver sees y x gain, gain = s_ref / scale, s_ref = the median scale of the 18 flat tuning
    sessions (U, tune50-export): the tuned penalty limits keep their average meaning, and weigh
    the same against every session's responses.  Also the 50 selected participants' real and test
    inputs when they exist, to show how far real scales reach beyond the tuning range."""
    from . import selection
    tune = pd.read_csv(TUNE50_CSV).session.tolist()
    names = [(f"{s}_{suffix}", "tune flat") for s in tune] + [(f"{s}_{suffix}s", "tune slow") for s in tune]
    if selection.SELECTION_CSV.exists():
        sel = pd.read_csv(selection.SELECTION_CSV)
        ids = sorted(sel.loc[sel.selected.astype(bool), "id"].astype(int))
        names += [(f"{k}{i}_{suffix}", kind) for k, kind in (("P", "real"), ("F", "test flat"), ("T", "test slow"))
                  for i in ids if (tx.OUT / f"{k}{i}_{suffix}.mat").exists()]
    out = pd.DataFrame([{"name": n, "kind": k, **input_scale(n)} for n, k in names])
    s_ref = float(out.query("kind == 'tune flat'").scale_uS.median())
    out["gain"] = [float(f"{v:.4g}") if np.isfinite(v) else 1.0 for v in s_ref / out.scale_uS]
    out["s_ref_uS"] = s_ref
    out.to_csv(SCALE_CSV, index=False)
    q = lambda v: v.quantile([0, 0.1, 0.5, 0.9, 1]).set_axis(["min", "q10", "median", "q90", "max"])
    print(f"s_ref = {s_ref:.4f} uS (median of the 18 flat tuning sessions)")
    print("typical response size (uS) per kind:")
    print(out.groupby("kind", sort=False).scale_uS.apply(q).unstack().to_string(float_format=lambda v: f"{v:.3f}"))
    print("gain per kind:")
    print(out.groupby("kind", sort=False).gain.apply(q).unstack().to_string(float_format=lambda v: f"{v:.2f}"))
    print("inputs with too few baseline peaks (whole input used):",
          out[out["from"] == "whole"].groupby("kind", sort=False).size().to_dict() or "none")
    lo, hi = out.query("kind == 'tune flat'").scale_uS.agg(["min", "max"])
    real = out.query("kind == 'real'")
    if len(real):
        print(f"real participants outside the tuning range {lo:.3f}-{hi:.3f} uS: "
              f"{int((real.scale_uS < lo).sum())} below, {int((real.scale_uS > hi).sum())} above, of {len(real)}")
    print("saved", SCALE_CSV.name)
    return out


def decay_check(setting=DECAY_SETTING):
    """Which tuning sessions get too-long decays?  One setting's decay error against session traits.

    Spearman correlation over the 18 tuning sessions, so one extreme session cannot drive it; with
    18 sessions it is a pointer, not a test.  Stated before looking: if the tonic is too stiff, the
    error grows with the source participant's tonic drift (+) and shrinks with the pulse rate (-),
    since quiet stretches are where uncaptured drift shows.
    """
    from scipy.stats import spearmanr
    s = pd.read_csv(TUNE_CSV).query("setting == @setting")
    c = pd.read_csv(ts.COHORT_CSV)[["session", "tau_decay", "rate_true_pm", "real_drift_uS", "real_level_uS"]]
    d = s.merge(c, on="session")
    traits = {"source tonic drift, uS": "real_drift_uS", "source tonic level, uS": "real_level_uS",
              "true pulse rate, /min": "rate_true_pm", "true decay, s": "tau_decay",
              "recall": "recall", "share of responses taken by the tonic": "phasic_taken"}
    rows = []
    for label, col in traits.items():
        rho, p = spearmanr(d[col], d.tau_decay_err_pct)
        rows.append({"setting": setting, "trait": label, "column": col, "spearman_rho": rho, "p_value": p})
    out = pd.DataFrame(rows)
    out.to_csv(DECAY_CSV, index=False)
    keep = ["session", "tau_group", "tau_decay", "tau_decay_fit", "tau_decay_err_pct", "rate_true_pm",
            "real_drift_uS", "real_level_uS", "recall", "phasic_taken"]
    d[keep].sort_values("tau_decay_err_pct").to_csv(DECAY_SESSIONS_CSV, index=False)
    print(d[keep].sort_values("tau_decay_err_pct").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{setting}: decay error (%) against each trait, Spearman over {len(d)} sessions")
    print(out[["trait", "spearman_rho", "p_value"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("saved", DECAY_CSV.name, DECAY_SESSIONS_CSV.name)
    return out


def nearest_gap(t, others, same=False):
    """Distance from each time in t to the nearest time in others.  same: t is part of others, so
    each time's zero distance to itself is skipped (true impulses are >= 1 s apart)."""
    if len(others) == 0:
        return np.full(len(t), np.inf)
    d = np.abs(t[:, None] - others[None, :])
    if same:
        d[d == 0] = np.inf
    return d.min(axis=1)


def error_check(setting=DECAY_SETTING, root=cfg.ROOT / "hpc" / "own" / "tune" / "output"):
    """Where do the remaining errors come from?  Every miss and false pulse of one setting, classified.

    Misses: true impulses >= 0.02 uS that no event matched (matched as recall_visible is), by size
    and by the gap to the nearest other true impulse.  False pulses: events matched to no true
    impulse (as precision counts them), by height and by the distance to the nearest true impulse.
    Read: false pulses near a true one are a response split in two (or mistimed), far ones are
    noise; misses with a close neighbour were merged, isolated small ones were lost in the noise.
    """
    from . import tbme_score as sc
    roles = split()
    rows, summary = [], []
    for r in roles[roles.role == "tune"].itertuples():
        name = f"{r.session}_own4"
        folder = root / setting
        tr = np.load(sc.truth_file(name))
        ev = pd.read_csv(folder / f"{name}_events.csv")
        win = pd.read_csv(folder / f"{name}_windows.csv")
        true_t, true_a = tr["ev_t"], tr["ev_a"]
        est_t, est_h = ev.time_s.to_numpy(), event_heights(ev, win)

        vis = true_a >= cfg.MIN_EVENT_uS
        vi, _ = sc.match(true_t[vis], est_t)
        vt, va = true_t[vis], true_a[vis]
        vgap = nearest_gap(vt, true_t, same=True)
        missed = np.ones(len(vt), bool)
        missed[vi] = False
        _, ei = sc.match(true_t, est_t)
        spurious = np.ones(len(est_t), bool)
        spurious[ei] = False
        fgap = nearest_gap(est_t, true_t)

        for k in np.flatnonzero(missed):
            rows.append({"session": r.session, "tau_group": r.tau_group, "kind": "miss",
                         "time_s": vt[k], "size_uS": va[k], "gap_s": vgap[k]})
        for k in np.flatnonzero(spurious):
            rows.append({"session": r.session, "tau_group": r.tau_group, "kind": "false",
                         "time_s": est_t[k], "size_uS": est_h[k], "gap_s": fgap[k]})
        for kind, size, gap, bad in (("miss", va, vgap, missed), ("false", est_h, fgap, spurious)):
            for big in (False, True):
                for near in (True, False):
                    m = ((size >= LARGE_uS) == big) & ((gap < CLOSE_S) == near)
                    summary.append({"session": r.session, "tau_group": r.tau_group, "kind": kind,
                                    "size": "large" if big else "small",
                                    "neighbour": "close" if near else "far",
                                    "n_all": int(m.sum()), "n_error": int((m & bad).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(ERRORS_CSV, index=False)
    s = pd.DataFrame(summary)
    s.to_csv(ERRORS_SUMMARY_CSV, index=False)

    for group in (None, "far", "moderate", "prior"):
        g = s if group is None else s[s.tau_group == group]
        t = g.groupby(["kind", "size", "neighbour"])[["n_all", "n_error"]].sum()
        t["error_rate"] = t.n_error / t.n_all.clip(lower=1)
        t["share_of_errors"] = t.n_error / t.groupby("kind").n_error.transform("sum").clip(lower=1)
        print(f"\n{setting}, {'all 18 sessions' if group is None else group + ' group'}"
              f"  (miss: n_all = visible true impulses, error_rate = missed share;"
              f" false: n_all = events, error_rate = false share)")
        print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nclose = a true impulse within {CLOSE_S:g} s; small = below {LARGE_uS:g} uS")
    print("saved", ERRORS_CSV.name, ERRORS_SUMMARY_CSV.name)
    return out
