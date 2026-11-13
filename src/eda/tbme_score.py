"""
Scoring a deconvolution run on the synthetic sessions against the truth that was put in.

For each session the events our solver returned (cluster output, <name>_events.csv,
<name>_windows.csv and <name>_signals.csv) are compared with the impulses that were put in
(data/tbme_synth/<name>.npz).  tuning.score runs this over every setting.

  matching     one-to-one, closest first, within 1 s (match below)
  recall       share of true impulses recovered; also over impulses >= 0.02 uS only, the
               smallest response peak detection could ever see (recall_visible, matched
               against those impulses alone).  True impulses are >= 1 s apart by construction,
               the paper's pruning window, so no closeness exclusion is needed
  precision    share of returned events that are real
  f1_visible   F1 from recall_visible and precision: the headline score
  timing       mean |time error| of matched pairs
  amplitude    Spearman rank correlation of matched pairs.  Ranks, because the solver's u is
               the input to its ODE and our a_i is a response height in uS: the units differ,
               the ordering should not
  tau          median over windows of the fitted tau_rise / tau_decay vs the truth
  fit          R2 of the reconstruction, from the windows table
  tonic        estimated vs true tonic, and phasic_taken: the share of each response the
               tonic absorbs (0 = clean split, 1 = the tonic takes the whole response)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as cfg
from . import tbme_synth as ts


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
    """S13_own -> data/tbme_synth/S13_own.npz: each input variant has its own (capped, filtered) truth."""
    base = name.split("_")[0]
    tag = next((t for t in ("_own", "_own4", "_own4s") if name.endswith(t)), "")
    return ts.TRUTH_DIR / f"{base}{tag}.npz"


def score_session(name, out_dir):
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
    # Visible: only true impulses >= 0.02 uS are targets, matched on their own so a tiny true
    # impulse cannot take the match from a larger one nearby.  Precision stays as above: an
    # event on a tiny true impulse is still real, not a false alarm.
    visible = true_a >= cfg.MIN_EVENT_uS
    vi, _ = match(true_t[visible], est_t)
    recall_vis = len(vi) / visible.sum() if visible.any() else np.nan
    return {
        "n_true": n_true, "n_est": n_est, "n_hit": n_hit,
        "recall": recall, "precision": precision,
        "f1": 2 * recall * precision / max(recall + precision, 1e-12),
        "n_visible": int(visible.sum()),
        "recall_visible": recall_vis,
        "f1_visible": 2 * recall_vis * precision / max(recall_vis + precision, 1e-12),
        "timing_err_s": np.abs(true_t[ti] - est_t[ei]).mean() if n_hit else np.nan,
        "amp_spearman": spearmanr(true_a[ti], est_a[ei])[0] if n_hit > 2 else np.nan,
        "tau_rise_fit": win.tau_rise.median(), "tau_decay_fit": win.tau_decay.median(),
        "tau_rise_err_pct": 100 * (win.tau_rise.median() / float(tr["tau_rise"]) - 1),
        "tau_decay_err_pct": 100 * (win.tau_decay.median() / float(tr["tau_decay"]) - 1),
        "tau_rise_spread": win.tau_rise.max() - win.tau_rise.min(),
        "tau_decay_spread": win.tau_decay.max() - win.tau_decay.min(),
        "r2_min": win.r2.min(), "r2_median": win.r2.median(),
        "windows_authors_rule": (win.selection == "authors").mean(),
        # share of windows whose GCV lambda stopped at its upper limit (fminbnd stops within its
        # log tolerance, so 0.95 x limit); NaN for runs before round 6, which did not log the limit
        "lambda2_at_cap": (win.lambda2 >= 0.95 * win.lambda2_cap).mean() if "lambda2_cap" in win else np.nan,
        "lambda1_at_cap": (win.lambda1 >= 0.95 * win.lambda1_cap).mean() if "lambda1_cap" in win else np.nan,
    }


FAST_HZ = 0.05                   # faster than a 20 s cycle: response time scale, not tonic


def fast_share(x, fs=ts.tx.FSY):
    """Share of a curve's variance (detrended) at frequencies above FAST_HZ."""
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return p[f > FAST_HZ].sum() / max(p[1:].sum(), 1e-30)


def tonic_metrics(name, out_dir):
    """Estimated vs true tonic of one synthetic run (see the docstring at the top).  All NaN when
    the run's per-sample signals are not there (the repository keeps them for a few runs only), so
    the pulse scores can still be computed."""
    f = out_dir / f"{name}_signals.csv"
    if not f.exists():
        return {k: np.nan for k in ("tonic_rmse_uS", "phasic_sd_uS", "phasic_taken", "fast_share_est",
                                    "fast_share_true", "fast_share_signal")}
    sig = pd.read_csv(f)
    tr = np.load(truth_file(name))
    true_tonic = np.interp(sig.time_s, tr["t2"], tr["tonic2"])
    true_phasic = np.interp(sig.time_s, tr["t2"], tr["phasic2"])
    tonic = sig.tonic.to_numpy() - np.median(sig.y - true_tonic - true_phasic)
    err = tonic - true_tonic
    fs = round(1 / float(np.median(np.diff(sig.time_s))))      # 2 Hz, or 4 Hz for _own4
    return {"tonic_rmse_uS": np.sqrt(np.mean(err ** 2)),
            "phasic_sd_uS": true_phasic.std(),
            "phasic_taken": np.cov(err, true_phasic)[0, 1] / max(true_phasic.var(), 1e-30),
            "fast_share_est": fast_share(tonic, fs), "fast_share_true": fast_share(true_tonic, fs),
            "fast_share_signal": fast_share(sig.y.to_numpy(), fs)}


def event_heights(ev, win, tu=0.25):
    """Each event's response height in uS: u x the peak of its window's ZOH response (own_matrices).

    The same conversion GCV-FOCUSS+ uses for its 0.02 uS threshold (u x max(B)).  u sits on a
    0.25 s grid in both inputs; each event takes the tau of the window whose centre is nearest,
    as the joined output keeps each window's middle half.
    """
    mid = ((win.t_start + win.t_end) / 2).to_numpy()
    w = np.abs(ev.time_s.to_numpy()[:, None] - mid[None, :]).argmin(axis=1)
    t = np.arange(0, 120, tu)
    peak = []
    for tr, td in zip(win.tau_rise, win.tau_decay):
        s = 1 - (td * np.exp(-t / td) - tr * np.exp(-t / tr)) / (td - tr)
        peak.append(np.diff(s).max())
    return ev.amplitude.to_numpy() * np.array(peak)[w]
