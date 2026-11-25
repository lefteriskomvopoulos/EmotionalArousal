"""
The runs after round 6, read from the cluster's output: the final run (the frozen setting on the
50 test sessions V and the 50 real participants P), the papers' re-synthesis test (R), their
figures, and the real participants' pulses as one table for stage 2.

  score_test50    test50-score: accuracy on V against the truth (data/own_test50_scores.csv)
  export_resynth  resynth-export: each real signal rebuilt from our deconvolution plus noise at its
                  own fit residual (R<pid>); score_resynth: resynth-score
  plot_final, plot_resynth   final-plots, resynth-plots: every signal before and after
  plot_fit_panels real50-panels: each real participant's raw signal, input, pulses and fit
  export_pulses   real-pulses: the 50 real participants' pulses and window fits, one table each
  export_extra    extra-export: real inputs of the other 47 who passed selection (exploratory);
                  export_extra_pulses: extra-pulses, their pulse tables (hpc/own/extra/output/)

Cluster output, per session <name> in hpc/own/final/output/<setting>/ (P, V) and
hpc/own/resynth/output/<setting>/ (R):
  <name>_events.csv   one row per pulse: time_s, amplitude (the impulse u)
  <name>_windows.csv  one row per 200 s window: tau_rise, tau_decay, lambda1, lambda2, r2, fit
                      selection rule, ...
  <name>_signals.csv  per sample: time_s, y (input), y_rec (fit), tonic, phasic, marker (block code)
"""
import numpy as np
import pandas as pd

from . import config as cfg
from . import tbme_export as tx
from . import tbme_synth as ts
from .generator import TEST6_CSV, selected_ids
from .preprocess import VARIANTS
from .tbme_score import event_heights

FINAL_OUT = cfg.ROOT / "hpc" / "own" / "final" / "output"
TEST6_SCORES_CSV = cfg.ROOT / "data" / "own_test50_scores.csv"
FINAL6_SETTING = "own4_p1000_k14_f3_s8"  # frozen after round 6 (make_own_bundle.sh)
RESYNTH_SEED = 9090
RESYNTH_CSV = cfg.ROOT / "data" / "resynth50_cohort.csv"
RESYNTH_SCORES_CSV = cfg.ROOT / "data" / "own_resynth50_scores.csv"
RESYNTH_OUT = cfg.ROOT / "hpc" / "own" / "resynth" / "output"
PULSES_CSV = cfg.ROOT / "data" / "real50_pulses.csv"
FITS_CSV = cfg.ROOT / "data" / "real50_windows.csv"


def score_test50(root=FINAL_OUT / FINAL6_SETTING):
    """test50-score: the frozen round-6 setting on the 50 test sessions (V), once, against their
    truth; overall, by response shape and by film group."""
    from . import tbme_score as sc
    c = pd.read_csv(TEST6_CSV)
    rows = []
    for r in c.itertuples():
        name = f"{r.session}_own4"
        res = sc.score_session(name, root)
        if res is None:
            print(f"  {name}: no output, skipped")
            continue
        rows.append({"session": r.session, "source_id": r.source_id, "group": r.group,
                     "shape": r.tau_group, **res, **sc.tonic_metrics(name, root)})
    out = pd.DataFrame(rows)
    out.to_csv(TEST6_SCORES_CSV, index=False)
    cols = ["f1_visible", "recall_visible", "precision", "f1_05", "recall_05", "precision_05", "timing_err_s", "amp_spearman",
            "tau_rise_err_pct", "tau_decay_err_pct", "phasic_taken", "windows_authors_rule", "lambda2_at_cap"]
    f = lambda v: f"{v:.3f}"
    print(f"{FINAL6_SETTING} on the {len(out)} test sessions (median)")
    print(out[cols].median().to_frame("all").T.to_string(float_format=f))
    print("by response shape:")
    print(out.groupby("shape")[cols].median().to_string(float_format=f))
    print("by film group:")
    print(out.groupby("group")[cols].median().to_string(float_format=f))
    print("saved", TEST6_SCORES_CSV.name)
    return out


def export_resynth(suffix="own4", root=FINAL_OUT / FINAL6_SETTING):
    """resynth-export: the papers' test (Amin & Faghih 2019, 2021), on our 50.  Each real
    participant's signal is rebuilt from our own deconvolution of it (final run: tonic + phasic,
    the joined reconstruction), white Gaussian noise is added, and the result R<pid>_<suffix> is
    deconvolved again and scored against our pulses (their heights in uS, event_heights; tau as
    the median of the windows).  At the solver rate, no preprocessing, as in the papers.
    The noise follows the papers' rule, not their number: they chose its level "to obtain
    comparable noise levels as in experimental data" (2019; 20-30 dB there, 25 dB in 2021).  Ours
    is each participant's own: the SD of their real signal minus our reconstruction.  (Their 25 dB
    against our rebuilt phasic part would have meant SD 0.0175 uS, about the smallest pulse.)
    Circular by construction: the truth is our own estimate, so pulses we missed on the real data
    are absent and our model's shape is exact; it measures self-consistency under noise, for
    comparison with the papers.  The main accuracy is test50-score."""
    fsy = VARIANTS[suffix][1]
    rows = []
    for pid in selected_ids():
        name, out = f"P{pid}_{suffix}", f"R{pid}_{suffix}"
        sig = pd.read_csv(root / f"{name}_signals.csv")
        ev = pd.read_csv(root / f"{name}_events.csv")
        win = pd.read_csv(root / f"{name}_windows.csv")
        phasic, rec = sig.phasic.to_numpy(), sig.y_rec.to_numpy()
        sd = float(np.std(sig.y.to_numpy() - rec))
        y = rec + sd * np.random.default_rng(RESYNTH_SEED + pid).standard_normal(len(rec))
        heights = event_heights(ev, win)
        tx.save(out, y, sig.time_s.to_numpy(), sig.marker.to_numpy(), 0, fs=fsy)
        np.savez_compressed(ts.TRUTH_DIR / f"{out}.npz", ev_t=ev.time_s.to_numpy(), ev_a=heights,
                            tonic2=sig.tonic.to_numpy(), phasic2=phasic,
                            tau_rise=float(win.tau_rise.median()), tau_decay=float(win.tau_decay.median()),
                            t2=sig.time_s.to_numpy(), source_id=pid)
        rows.append({"session": f"R{pid}", "source_id": pid, "noise_sd_uS": sd,
                     "snr_db": float(10 * np.log10(np.mean(phasic ** 2) / sd ** 2)) if sd > 0 else np.nan,
                     "n_true": len(ev),
                     "true_med_uS": float(np.median(heights)) if len(heights) else np.nan,
                     "tau_rise": float(win.tau_rise.median()), "tau_decay": float(win.tau_decay.median()),
                     "r2_real": 1 - np.sum((sig.y - rec) ** 2) / np.sum((sig.y - sig.y.mean()) ** 2)})
    from . import selection
    c = pd.DataFrame(rows)
    sel = pd.read_csv(selection.SELECTION_CSV)[["id", "group"]]
    c = c.merge(sel.rename(columns={"id": "source_id"}), on="source_id", how="left")
    c.to_csv(RESYNTH_CSV, index=False)
    print(f"rebuilt {len(c)} signals from {root.name}; noise at each participant's own fit residual:"
          f" SD median {c.noise_sd_uS.median():.4f} uS ({c.noise_sd_uS.min():.4f}-{c.noise_sd_uS.max():.4f}),"
          f" i.e. {c.snr_db.median():.0f} dB against the phasic part ({c.snr_db.min():.0f}-{c.snr_db.max():.0f});"
          f" true pulses median {c.n_true.median():.0f} per session, height median {c.true_med_uS.median():.3f} uS;"
          f" fit of the real signals R2 median {c.r2_real.median():.3f}")
    print(f"saved R*_{suffix} in {tx.OUT.name}, truth in {ts.TRUTH_DIR.name}; {RESYNTH_CSV.name}")
    return c


def score_resynth(root=RESYNTH_OUT / FINAL6_SETTING):
    """resynth-score: the frozen setting on the 50 rebuilt signals (R), against our own pulses."""
    from . import tbme_score as sc
    c = pd.read_csv(RESYNTH_CSV)
    rows = []
    for r in c.itertuples():
        name = f"{r.session}_own4"
        res = sc.score_session(name, root)
        if res is None:
            print(f"  {name}: no output, skipped")
            continue
        rows.append({"session": r.session, "source_id": r.source_id, "group": r.group,
                     **res, **sc.tonic_metrics(name, root)})
    out = pd.DataFrame(rows)
    out.to_csv(RESYNTH_SCORES_CSV, index=False)
    cols = ["f1_visible", "recall_visible", "precision", "f1_05", "recall_05", "precision_05", "timing_err_s", "amp_spearman",
            "tau_rise_err_pct", "tau_decay_err_pct", "r2_median"]
    f = lambda v: f"{v:.3f}"
    print(f"re-synthesis (noise at each participant's fit residual), {FINAL6_SETTING}, {len(out)} sessions (median):")
    print(out[cols].median().to_frame("all").T.to_string(float_format=f))
    print("by film group:")
    print(out.groupby("group")[cols].median().to_string(float_format=f))
    print("saved", RESYNTH_SCORES_CSV.name)
    return out


def _deconv_item(name, root, title, truth=None, split_uS=None):
    """One deconvolved session for figures.deconv_pages: input, reconstruction, tonic, and its
    pulses: with truth, found / missed (true >= 0.02 uS, none within 1 s) / false (estimated, no
    true one within 1 s), matched as tbme_score does; without, the estimated pulses, in two rows
    by response height (event_heights) when split_uS is given."""
    from . import figures
    from . import tbme_score as sc
    sig = pd.read_csv(root / f"{name}_signals.csv")
    ev = pd.read_csv(root / f"{name}_events.csv")
    est_t = ev.time_s.to_numpy()
    if truth is None and split_uS is not None:
        h = np.asarray(event_heights(ev, pd.read_csv(root / f"{name}_windows.csv")), float)
        pulses = [(f"pulses ≥ {split_uS:g} µS", est_t[h >= split_uS], figures.COLOR_EVENTS),
                  (f"pulses < {split_uS:g} µS", est_t[h < split_uS], "0.6")]
    elif truth is None:
        pulses = [("estimated pulses", est_t, figures.COLOR_EVENTS)]
    else:
        true_t, true_a = np.asarray(truth["ev_t"]), np.asarray(truth["ev_a"])
        ti, ei = sc.match(true_t, est_t)
        hit, used = np.zeros(len(true_t), bool), np.zeros(len(est_t), bool)
        hit[ti], used[ei] = True, True
        pulses = [("found", true_t[hit], figures.COLOR_EVENTS),
                  ("missed", true_t[~hit & (true_a >= cfg.MIN_EVENT_uS)], figures.COLOR_MISSED),
                  ("false", est_t[~used], figures.COLOR_FALSE)]
    return {"name": name.split("_")[0], "title": title, "t": sig.time_s.to_numpy(), "y": sig.y.to_numpy(),
            "y_rec": sig.y_rec.to_numpy(), "tonic": sig.tonic.to_numpy(), "mk": sig.marker.to_numpy(),
            "pulses": pulses}


def plot_final(suffix="own4", root=FINAL_OUT / FINAL6_SETTING):
    """final-plots: every signal of the final run before and after deconvolution.
    Test sessions (V): found, missed and false pulses against their truth, worst F1 first.
    Real participants (P): their estimated pulses.
    -> results/figures/test50_deconv.pdf, real50_deconv.pdf, and PNGs in test50_deconv/, real50_deconv/"""
    from . import figures
    s = pd.read_csv(TEST6_SCORES_CSV).sort_values("f1_visible")
    items = []
    for r in s.itertuples():
        name = f"{r.session}_{suffix}"
        tr = np.load(ts.TRUTH_DIR / f"{name}.npz")
        title = (f"{r.session} (test, from P{r.source_id}, {r.group} films, {r.shape} shape: true tau"
                 f" {float(tr['tau_rise']):.2f} / {float(tr['tau_decay']):.2f} s, fitted {r.tau_rise_fit:.2f} /"
                 f" {r.tau_decay_fit:.2f} s)\nF1 (visible) {r.f1_visible:.3f}, recall {r.recall_visible:.3f},"
                 f" precision {r.precision:.3f}, timing error {r.timing_err_s:.2f} s, size rank"
                 f" {r.amp_spearman:.3f}; fit R2 median {r.r2_median:.3f}")
        items.append(_deconv_item(name, root, title, tr))
    figures.deconv_pages(items, "test50_deconv", "test50_deconv.pdf")

    from . import selection
    sel = pd.read_csv(selection.SELECTION_CSV)
    sel = sel[sel.selected.astype(bool)].sort_values(["group", "id"])
    items = []
    for r in sel.itertuples():
        name = f"P{int(r.id)}_{suffix}"
        win = pd.read_csv(root / f"{name}_windows.csv")
        n = len(pd.read_csv(root / f"{name}_events.csv"))
        minutes = (win.t_end.max() - win.t_start.min()) / 60
        title = (f"P{int(r.id)} (real, {r.group} films): {n} pulses ({n / minutes:.1f} per min), fitted tau"
                 f" {win.tau_rise.median():.2f} / {win.tau_decay.median():.2f} s, fit R2 median"
                 f" {win.r2.median():.3f}, lambda2 median {win.lambda2.median():.2e}"
                 f" ({(win.lambda2 >= 0.95 * win.lambda2_cap).mean():.0%} of windows at its limit)")
        items.append(_deconv_item(name, root, title, split_uS=0.05))  # stage 2's main threshold
    figures.deconv_pages(items, "real50_deconv", "real50_deconv.pdf")


def plot_resynth(suffix="own4", root=RESYNTH_OUT / FINAL6_SETTING):
    """resynth-plots: the rebuilt signals (R) before and after deconvolution, found / missed /
    false against our own pulses, worst F1 first.  -> results/figures/resynth50_deconv.pdf"""
    from . import figures
    s = pd.read_csv(RESYNTH_SCORES_CSV).sort_values("f1_visible")
    items = []
    for r in s.itertuples():
        name = f"{r.session}_{suffix}"
        tr = np.load(ts.TRUTH_DIR / f"{name}.npz")
        title = (f"{r.session} (rebuilt from our deconvolution of P{r.source_id}, {r.group} films)\n"
                 f"F1 (visible) {r.f1_visible:.3f}, recall {r.recall_visible:.3f}, precision {r.precision:.3f},"
                 f" timing error {r.timing_err_s:.2f} s, size rank {r.amp_spearman:.3f}")
        items.append(_deconv_item(name, root, title, tr))
    figures.deconv_pages(items, "resynth50_deconv", "resynth50_deconv.pdf")


def _fit_item(pid, group, suffix, root):
    """One real participant for figures.fit_pages: raw recording (every 10th sample), the solver's
    input, reconstruction and tonic, and each pulse's height in uS (event_heights)."""
    from .dataset import load
    name = f"P{pid}_{suffix}"
    sig = pd.read_csv(root / f"{name}_signals.csv")
    ev = pd.read_csv(root / f"{name}_events.csv")
    win = pd.read_csv(root / f"{name}_windows.csv")
    raw = load(pid, every=10)
    y, rec = sig.y.to_numpy(), sig.y_rec.to_numpy()
    h = np.asarray(event_heights(ev, win), float)
    minutes = (sig.time_s.iloc[-1] - sig.time_s.iloc[0]) / 60
    r2 = 1 - np.sum((y - rec) ** 2) / np.sum((y - y.mean()) ** 2)
    big = int((h >= 0.05).sum())
    title = (f"P{pid} (real, {group} films): {len(h)} pulses ({len(h) / minutes:.1f} per min), {big} of them"
             f" ≥ 0.05 µS ({big / minutes:.1f} per min)\nfit R² {r2:.4f} (window median {win.r2.median():.4f}),"
             f" residual SD {np.std(y - rec):.4f} µS; fitted τ rise / decay {win.tau_rise.median():.2f} /"
             f" {win.tau_decay.median():.2f} s (window medians)")
    print(f"  P{pid}: {len(h)} pulses, R2 {r2:.4f}")
    return {"title": title, "raw_t": raw["t"], "raw_x": raw["eda"], "t": sig.time_s.to_numpy(), "y": y,
            "y_rec": rec, "tonic": sig.tonic.to_numpy(), "mk": sig.marker.to_numpy(),
            "ev_t": ev.time_s.to_numpy(), "ev_h": h}


def plot_fit_panels(suffix="own4", root=FINAL_OUT / FINAL6_SETTING):
    """real50-panels: each real participant's fit, to judge by eye, one page each, by film group:
    1 the raw recording, 2 the solver's input, 3 the pulses with their heights in uS (>= 0.05 uS,
    the main analysis, apart from the rest), 4 the reconstruction over the input.  Whole analysed
    segment and a 120 s zoom from the first film's start.  Reads the final run's output and the raw
    recordings (data/raw/Study6); writes nothing else.  -> results/figures/real50_fit_panels.pdf"""
    from . import figures, selection
    sel = pd.read_csv(selection.SELECTION_CSV)
    sel = sel[sel.selected.astype(bool)].sort_values(["group", "id"])
    figures.fit_pages((_fit_item(int(r.id), r.group, suffix, root) for r in sel.itertuples()),
                      "real50_fit_panels.pdf")


def export_pulses(suffix="own4", root=FINAL_OUT / FINAL6_SETTING, ids=None,
                  pulses_csv=PULSES_CSV, fits_csv=FITS_CSV):
    """real-pulses: the frozen setting's pulses on the 50 real participants, the input of stage 2.
    With ids, root and the two paths: the same tables for other participants (extra-pulses).

    data/real50_pulses.csv, one row per pulse:
      id, group      participant and film group (selection.csv)
      time_s         on the recording's clock, s
      u              the impulse, as the solver returns it (events.csv amplitude)
      height_uS      the response it makes: u x the peak of its window's response (event_heights);
                     the solver keeps only pulses >= 0.02 uS; stage 2 counts those >= 0.05 uS
                     (main analysis; all of them as the secondary check)
      marker         block code at that time: -1 baseline, 1xx neutral, 2xx negative, 3xx positive
                     film, empty between blocks
      block          baseline / film / other;  valence  neutral / negative / positive for films
      window         the 200 s window whose fit it came from
    data/real50_windows.csv, one row per window: id, group and the window's fit (windows.csv).
    Reads only the final run's output, so it needs no MATLAB, cluster or raw data."""
    from . import selection
    sel = pd.read_csv(selection.SELECTION_CSV)
    keep = sel.selected.astype(bool) if ids is None else sel.id.isin(ids)
    sel = sel[keep].sort_values("id")
    pulses, fits = [], []
    for r in sel.itertuples():
        pid, name = int(r.id), f"P{int(r.id)}_{suffix}"
        ev = pd.read_csv(root / f"{name}_events.csv")
        win = pd.read_csv(root / f"{name}_windows.csv")
        sig = pd.read_csv(root / f"{name}_signals.csv", usecols=["time_s", "marker"])
        t = ev.time_s.to_numpy()
        mid = ((win.t_start + win.t_end) / 2).to_numpy()
        w = np.abs(t[:, None] - mid[None, :]).argmin(axis=1) if len(t) else np.zeros(0, int)
        k = np.clip(np.searchsorted(sig.time_s.to_numpy(), t), 0, len(sig) - 1)
        mk = sig.marker.to_numpy(float)[k]
        film = (mk >= 100) & (mk < 400)
        pulses.append(pd.DataFrame({
            "id": pid, "group": r.group, "time_s": t, "u": ev.amplitude.to_numpy(),
            "height_uS": event_heights(ev, win), "marker": pd.array(mk, dtype="Float64").astype("Int64"),
            "block": np.where(mk == -1, "baseline", np.where(film, "film", "other")),
            "valence": np.where(film, np.select([mk < 200, mk < 300], ["neutral", "negative"], "positive"), ""),
            "window": win.window.to_numpy()[w]}))
        fits.append(win.assign(id=pid, group=r.group))
    p = pd.concat(pulses, ignore_index=True)
    f = pd.concat(fits, ignore_index=True)
    lead = ["id", "group"]
    f = f[lead + [c for c in f.columns if c not in lead]]
    p.to_csv(pulses_csv, index=False)
    f.to_csv(fits_csv, index=False)
    minutes = (f.groupby("id").t_end.max() - f.groupby("id").t_start.min()) / 60
    rate = p.groupby("id").size().reindex(minutes.index, fill_value=0) / minutes
    # overall only: rates by film group or block are what H1-H3 test, so they are not printed here
    print(f"{len(p)} pulses from {p.id.nunique()} participants ({FINAL6_SETTING}); per minute, median"
          f" {rate.median():.1f} ({rate.min():.1f}-{rate.max():.1f})")
    print(f"height median {p.height_uS.median():.3f} uS, {(p.height_uS >= 0.05).mean():.0%} >= 0.05 uS")
    print("saved", pulses_csv.name, fits_csv.name)
    return p, f


# The other 47 (exploratory, for stage 2): the participants who passed both selection stages but
# were not drawn into the 50 (selection.csv: stage2_pass and not selected).  The frozen setting,
# unchanged, run once on their real inputs.  Not part of the reported results or of H1-H6.
EXTRA_OUT = cfg.ROOT / "hpc" / "own" / "extra" / "output"
EXTRA_PULSES_CSV = cfg.ROOT / "data" / "extra47_pulses.csv"
EXTRA_FITS_CSV = cfg.ROOT / "data" / "extra47_windows.csv"


def extra_ids():
    """The 47: passed quality (stage 1) and the baseline rule (stage 2), not drawn (stage 3)."""
    from . import selection
    sel = pd.read_csv(selection.SELECTION_CSV)
    return sorted(sel.loc[sel.stage2_pass.astype(bool) & ~sel.selected.astype(bool), "id"].astype(int))


def export_extra(suffix="own4"):
    """extra-export: the real inputs P<pid>_<suffix> of the 47, through the same chain and the same
    lines as final-export's real inputs (prepare, then tbme_export.save).  Writes nothing else.  An
    input that already exists is left alone if it is the same signal; if it differs, this stops,
    because a kept run may have been made from it."""
    from scipy.io import loadmat
    from .dataset import load
    from .preprocess import RAW_FS, prepare, segment_of
    cutoff, fsy = VARIANTS[suffix]
    ids = extra_ids()
    written, same = 0, 0
    for pid in ids:
        name = f"P{pid}_{suffix}"
        d = load(pid, every=1)
        pr = prepare(d["eda"], d["t"], RAW_FS, segment_of(d["marker"]), cutoff, fsy)
        path = tx.OUT / f"{name}.mat"
        if path.exists():
            old = loadmat(path)["y"].ravel()
            if len(old) == len(pr["y"]) and np.allclose(old, pr["y"], rtol=0, atol=1e-12):
                same += 1
                print(f"  {name}: exists, same signal, kept")
                continue
            raise RuntimeError(f"{path} exists and differs from the rebuilt input; not overwritten")
        tx.save(name, pr["y"], pr["t2"], d["marker"][pr["idx"]], int(pr["gaps"].sum()), fs=fsy)
        written += 1
        print(f"  {name}: {len(pr['y'])} samples")
    print(f"{len(ids)} participants: {written} inputs written, {same} already there -> {tx.OUT}")


def export_extra_pulses(suffix="own4"):
    """extra-pulses: the 47's pulses and window fits, in the format of real-pulses
    -> data/extra47_pulses.csv, data/extra47_windows.csv"""
    return export_pulses(suffix, EXTRA_OUT / FINAL6_SETTING, extra_ids(), EXTRA_PULSES_CSV, EXTRA_FITS_CSV)


BLOCK_RATES_CSV = cfg.ROOT / "data" / "real50_block_rates.csv"
BLOCK_KINDS = ("baseline", "neutral", "negative", "positive")


def _block_kind(mk):
    """Block code -> baseline / neutral (1xx) / negative (2xx) / positive (3xx) film / other."""
    mk = np.asarray(mk, float)
    out = np.full(len(mk), "other", dtype=object)
    out[mk == -1] = "baseline"
    for lo, kind in ((100, "neutral"), (200, "negative"), (300, "positive")):
        out[(mk >= lo) & (mk < lo + 100)] = kind
    return out


def block_rates(suffix="own4", root=FINAL_OUT / FINAL6_SETTING):
    """block-rates: descriptive pulse rates and sizes of the 50 real participants by block (baseline, and
    neutral / negative / positive films), per film group.  A first descriptive look at the
    quantities H1-H3 are about, taken on 2026-11-14 after the solver, the selection and stage 2's
    threshold were frozen (docs/EXPERIMENTS.md): no test, and nothing in stage 1 changes after it.
    Main: pulses >= 0.05 uS; secondary: all returned pulses (>= 0.02 uS).  Minutes per block from
    each participant's final-run signals (4 Hz block codes).
    -> data/real50_block_rates.csv (one row per participant and block)"""
    from . import selection
    sel = pd.read_csv(selection.SELECTION_CSV)
    sel = sel[sel.selected.astype(bool)].sort_values("id")
    pulses = pd.read_csv(PULSES_CSV)
    rows = []
    for r in sel.itertuples():
        pid = int(r.id)
        sig = pd.read_csv(root / f"P{pid}_{suffix}_signals.csv", usecols=["time_s", "marker"])
        kind = _block_kind(sig.marker.to_numpy(float))
        dt = float(np.median(np.diff(sig.time_s.to_numpy())))
        p = pulses[pulses.id == pid]
        k = np.clip(np.searchsorted(sig.time_s.to_numpy(), p.time_s.to_numpy()), 0, len(sig) - 1)
        pk = kind[k]
        h = p.height_uS.to_numpy()
        big = h >= ts.SPLIT_uS
        for b in BLOCK_KINDS:
            minutes = (kind == b).sum() * dt / 60
            if minutes <= 0:
                continue
            here = pk == b
            n_all, n_05 = int(here.sum()), int((here & big).sum())
            med = lambda v: float(np.median(v)) if len(v) else np.nan
            rows.append({"id": pid, "group": r.group, "block": b, "minutes": minutes, "n_all": n_all,
                         "n_05": n_05, "rate_all_pm": n_all / minutes, "rate_05_pm": n_05 / minutes,
                         # size: the median pulse height, and the summed heights per minute (rate x size)
                         "size_05_uS": med(h[here & big]), "size_all_uS": med(h[here]),
                         "drive_05_uS_pm": h[here & big].sum() / minutes, "drive_all_uS_pm": h[here].sum() / minutes})
    out = pd.DataFrame(rows)
    out.to_csv(BLOCK_RATES_CSV, index=False)

    f = lambda v: f"{v:.1f}"
    for col, label in (("rate_05_pm", f">= {ts.SPLIT_uS} uS (main)"), ("rate_all_pm", "all, >= 0.02 uS (secondary)")):
        t = out.pivot_table(index="group", columns="block", values=col, aggfunc="median")
        t = t[[b for b in BLOCK_KINDS if b in t.columns]]
        print(f"\npulses per minute, {label}: median over participants, by film group and block")
        print(t.to_string(float_format=f))
    n = out.pivot_table(index="group", columns="block", values="id", aggfunc="count").fillna(0).astype(int)
    print("\nparticipants with each block:")
    print(n[[b for b in BLOCK_KINDS if b in n.columns]].to_string())
    m = out.pivot_table(index="group", columns="block", values="minutes", aggfunc="median")
    print("\nminutes per block, median:")
    print(m[[b for b in BLOCK_KINDS if b in m.columns]].to_string(float_format=f))
    for col, label in (("size_05_uS", f"median pulse height, uS, pulses >= {ts.SPLIT_uS} uS"),
                       ("drive_05_uS_pm", f"summed pulse heights per minute (rate x size), uS/min, pulses >= {ts.SPLIT_uS} uS")):
        t = out.pivot_table(index="group", columns="block", values=col, aggfunc="median")
        t = t[[b for b in BLOCK_KINDS if b in t.columns]]
        print(f"\n{label}: median over participants, by film group and block")
        print(t.to_string(float_format=lambda v: f"{v:.3f}" if col.startswith("size") else f"{v:.2f}"))
    print(f"\nwithin participant, films (mean of their film kinds) / baseline, pulses >= {ts.SPLIT_uS} uS:"
          " median by group")
    ratios = {}
    for col, name in (("rate_05_pm", "rate"), ("size_05_uS", "size"), ("drive_05_uS_pm", "rate x size")):
        w = out.pivot_table(index=["id", "group"], columns="block", values=col)
        films = [b for b in ("neutral", "negative", "positive") if b in w.columns]
        ratios[name] = (w[films].mean(axis=1) / w["baseline"]).replace([np.inf, -np.inf], np.nan)
    print(pd.DataFrame(ratios).groupby(level="group").median().to_string(float_format=lambda v: f"{v:.2f}"))
    print("descriptive only: no test; block lengths, film order and habituation differ between blocks."
          "  Size was added after rate was seen (2026-11-14), recorded in docs/EXPERIMENTS.md")
    print("saved", BLOCK_RATES_CSV.name)
    return out
