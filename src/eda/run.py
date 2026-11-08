"""
Command-line entry point.  From the project root:

    PYTHONPATH=src python -m eda.run <command> [participant_id]

docs/EXPERIMENTS.md explains what each test found.  Commands are listed in the order of that story.

Data
    inspect            what is in the raw folder: files, channels, one block timeline
    quality            screen all 187 participants -> data/quality_study6.csv
    plot               raw EDA + affect slider for the chosen participants
    profile            model-free description of every usable participant + their 2 Hz inputs
                       -> data/profile_study6*.csv/.npz, data/tbme_input/P*.mat

Our own deconvolution (src/own/), branch own-deconvolution
    raw-atlas          all 119 participants + 18 tuning sessions, unprocessed, ~6 min
                       -> results/figures/raw_atlas_*.png, data/raw_overview.csv
    preprocess-stages  every preprocessing step, 5 real + 5 tuning sessions, ~2 min
                       -> results/figures/stages_*.png, data/preprocess_stages.csv
    own-check          all 119 through prepare: roughness, jumps, gaps + the filter's response, ~8 min
                       -> data/input_check.csv, data/filter_response.csv, results/figures/filter_response.png
    own-score          the cluster runs in hpc/own/tune/output/<setting>, against the truth
                       -> data/own_tune_scores.csv
    own-export         needs own-check.  The inputs through own_preprocess.prepare: 36 synthetic + 5 real
                       -> data/tbme_input/*_own.mat, data/tbme_synth/*_own.npz
    own4-export        the same at 1 Hz low-pass, 4 Hz, noise matched to the sources at that rate
                       -> data/input_check_own4.csv, data/tbme_input/*_own4.mat, data/tbme_synth/*_own4.npz

Current pipeline: Amin & Faghih (IEEE TBME 2021), authors' MATLAB solver unmodified, see src/tbme/
    paper-export       the paper's preprocessing on 5 real + 8 synthetic -> data/tbme_input/*_pp.mat
    paper-test         earlier inputs vs the paper's preprocessing -> data/tbme_paper_*.csv
    tbme-real [pid]    real participants' cluster output (hpc/real/output): summary + figures
                       -> data/tbme_real_summary.csv, results/figures/tbme_real_P*.png
    state   [pid]      stage 2, arousal state; still reads archive/deconvolution.py + figure

Validating the published method on synthetic sessions (cluster output in hpc/synth/output)
    tbme-export [pid]  2 Hz input file for one participant, or all usable ones, no filtering
    synth-tbme         36 synthetic sessions drawn from the profiles (tbme_synth.py)
                       -> data/tbme_input/S*.mat, data/tbme_synth/, data/tbme_synth_*.csv
    score-tbme         score the cluster output against the truth -> data/tbme_synth_scores.csv
    synth-shift        low-level synthetic sessions shifted to minimum 0 and 2 uS (S<k>_m0/_m2)
    score-shift        score them against the unshifted runs -> data/tbme_synth_shift_scores.csv
    false-events       what the false events are (size, kind)  -> data/tbme_synth_false_events.csv
    cutoff-test        size cut-off / merging, chosen on half    -> data/tbme_synth_cutoff.csv
    importance-test    size cut-off vs "does the pulse matter" -> data/tbme_synth_importance*.csv

The tau problem
    window-test        far-group sessions at 200 / 400 / 800 s -> data/tbme_synth_window_scores.csv
    synth-rate         4 far + 4 prior sessions rebuilt at 4 Hz (S<k>_f4)
    rate-test          2 Hz vs 4 Hz, tau and recovery -> data/tbme_synth_rate_scores.csv
    tail-test          tail-pulse fingerprint, synthetic + real -> data/tbme_tail_fingerprint.csv
    tonic-test         estimated vs true tonic, synthetic + real -> data/tbme_tonic_check.csv
    fixed-tau-test     tau forced to the truth / prior -> data/tbme_synth_fixed_tau.csv
    tau-measure        tau from isolated responses, with held-out checks
                       -> data/tbme_tau_measure.csv, data/tbme_tau_responses.csv
    tau-shared         one tau per person from all isolated responses, post-hoc retry
                       -> data/tbme_tau_shared.csv, data/tbme_tau_shared_responses.csv

Preprocessing checks
    preprocess-view    raw vs 2 Hz input, 5 real + 5 synthetic, and real vs synthetic
                       -> results/figures/preprocess_*.png, data/preprocess_compare.csv
    paper-check        the authors' own 100 Hz input vs ours
                       -> data/paper_input_*.csv, results/figures/paper_input_spectrum.png

Superseded, code in src/eda/archive/.  Our own deconvolution:
    deconv  [pid]      deconvolution over a whole session + figure
    window  [pid]      one 120 s window through every stage + figure
    sweep   [pid]      lambda sweep on one participant
    sweep-all          lambda sweep on weak and strong responders
    kernels            kernel identifiability across all usable participants
    tonic              tonic-window sweep
    synth              stage 1 on synthetic sessions with a known answer -> validation table
    synth-lam          the same, at several lambda values, with and without the true tonic/kernel
    synth-fix          the same, comparing the candidate fixes (kernel cost, debiasing, tonic)
    synth-robust       current pipeline +/- debiasing across burst sizes and lambda values
    synth-gcv          lambda chosen per session by GCV, alternating with the kernel fit
    synth-scale        lambda proportional to each session's burst scale, simple + realistic sets

Superseded: cvxEDA (Greco et al. 2016), authors' code unmodified
    cvx-smoke          one synthetic session: checks the install, times one 20-min solve
    cvx-synth          all synthetic sessions, scored like synth-scale
    cvx     [pid]      one real participant, cvxEDA next to our pipeline + figures
    cvx-params         synthetic sessions under each cvx.CONFIGS setting + the true tau
    cvx-real           two usable participants per group under each cvx.CONFIGS setting
    cvx-alpha          synthetic sessions across cvxEDA's alpha, default and true tau
    cvx-alpha-real     check only: real impulse rates across alpha, 8 participants
    cvx-tau            per-session tau by an outer loop around cvxEDA, synthetic sessions
"""
import sys

import numpy as np

from . import config as cfg
from .archive import cvx, experiments, validation
from .archive import deconvolution as dec
from . import figures
from . import state as st
from . import cohort_profile
from . import tbme_synth
from . import tbme_score
from . import tau_measure
from . import tau_shared
from . import own_preprocess
from . import preprocess_view
from . import tbme_real
from . import tbme_export
from .dataset import baseline_mask, film_mask, load, read_header
from .quality import choose_participants, run_screen


def cmd_inspect():
    """Print the raw folder's contents, the channel header, and one block timeline."""
    files = sorted(cfg.RAW.glob("*.csv"))
    print(f"{len(files)} csv files, {len(list(cfg.RAW.glob('*_All.csv')))} participants")

    first = sorted(cfg.RAW.glob("*_All.csv"))[0]
    for k, v in read_header(first).items():
        if k.startswith(("Channel", "Data_Unit", "Data_Sample", "Data_Device")):
            print(f"  {k:17s} {v}")

    d = load(first.name.split("_")[1][1:])
    print(f"\nblock timeline of {first.name}:")
    for i in np.flatnonzero(d["marker"][1:] != d["marker"][:-1]) + 1:
        print(f"  {d['t'][i]:7.1f} s  ->  {d['marker'][i]}")


def cmd_plot():
    """Raw EDA and slider for a couple of usable participants per group."""
    records = [(pid, group, load(pid)) for pid, group in choose_participants()]
    for pid, group, d in records:
        print(f"P{pid} {group}: {d['t'][-1] / 60:.1f} min, "
              f"EDA {np.nanmin(d['eda']):.2f}–{np.nanmax(d['eda']):.2f} µS, "
              f"missing {np.isnan(d['eda']).mean() * 100:.1f}%")
    figures.raw_signals(records)


def cmd_deconv(pid=None):
    """Deconvolve a whole session and report per-block impulse statistics."""
    pid = pid or choose_participants()[0][0]
    d = dec.run(pid, load)
    t, mk, ev_t, ev_a = d["t"], d["marker"], d["ev_t"], d["ev_a"]

    minutes = (t[-1] - t[0]) / 60
    print(f"P{pid}: tau_rise {d['tau_rise']:.2f} s, tau_decay {d['tau_decay']:.2f} s | "
          f"{len(ev_t)} impulses ({len(ev_t) / minutes:.1f}/min) | fit R2 {d['r2']:.3f}")

    for name, mask in (("baseline", baseline_mask(mk)), ("films", film_mask(mk))):
        if mask.sum():
            sel = np.isin(((ev_t - t[0]) * cfg.DFS).astype(int), np.flatnonzero(mask))
            print(f"   {name:9s} {sel.sum():4d} impulses, "
                  f"{sel.sum() / (mask.sum() / cfg.DFS / 60):5.1f}/min, "
                  f"mean amplitude {ev_a[sel].mean() if sel.any() else 0:.3f}")

    figures.deconvolution_session(pid, d)


def cmd_cvx(pid=99):
    """cvxEDA as published on one participant, side by side with our pipeline."""
    import time
    for name, fn, tag in (("ours", dec.run, ""), ("cvxEDA", cvx.run, "_cvxeda")):
        t0 = time.time()
        d = fn(pid, load)
        t, mk, ev_t, ev_a = d["t"], d["marker"], d["ev_t"], d["ev_a"]
        print(f"P{pid} {name:6s}: tau {d['tau_rise']:.2f}/{d['tau_decay']:.2f} s | "
              f"{len(ev_t)} impulses ({len(ev_t) / ((t[-1] - t[0]) / 60):.1f}/min) | "
              f"R2 {d['r2']:.3f} | {time.time() - t0:.0f} s")
        for block, mask in (("baseline", baseline_mask(mk)), ("films", film_mask(mk))):
            if mask.sum():
                sel = np.isin(((ev_t - t[0]) * cfg.DFS).astype(int), np.flatnonzero(mask))
                print(f"   {block:9s} {sel.sum() / (mask.sum() / cfg.DFS / 60):5.1f}/min, "
                      f"median amplitude {np.median(ev_a[sel]) if sel.any() else 0:.3f} uS")
        figures.deconvolution_session(pid, d, tag=tag)


def cmd_cvx_real():
    """cvxEDA under each setting in cvx.CONFIGS on two usable participants per group."""
    import pandas as pd
    rows = []
    for pid, group in choose_participants():
        d = dec.to_deconv_rate(load(pid))
        t, mk = d["t"], d["marker"]
        masks = {"baseline": baseline_mask(mk), "films": film_mask(mk)}
        for conf, kw in cvx.CONFIGS.items():
            r = cvx.decompose(d["eda"], t0=t[0], **kw)
            idx = ((r["ev_t"] - t[0]) * cfg.DFS).astype(int)
            row = {"pid": pid, "group": group, "config": conf, "r2": r["r2"]}
            for block, mask in masks.items():
                sel = np.isin(idx, np.flatnonzero(mask))
                row[f"{block}_per_min"] = sel.sum() / (mask.sum() / cfg.DFS / 60)
                row[f"{block}_median_uS"] = np.median(r["ev_a"][sel]) if sel.any() else np.nan
            rows.append(row)
        print(f"P{pid} {group} done")

    v = pd.DataFrame(rows)
    out = cfg.ROOT / "data" / "cvxeda_real_configs.csv"
    v.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 3):
        print(f"\n{v['pid'].nunique()} participants -> {out.name}, median per setting:")
        print(v.groupby("config", sort=False).median(numeric_only=True).drop(columns="pid"))


def cmd_cvx_alpha_real():
    """Check only, never used to choose: impulse rates on real data across cvxEDA's alpha."""
    import pandas as pd
    rows = []
    for pid, group in choose_participants():
        d = dec.to_deconv_rate(load(pid))
        t, mk = d["t"], d["marker"]
        masks = {"baseline": baseline_mask(mk), "films": film_mask(mk)}
        for alpha in validation.ALPHAS:
            r = cvx.decompose(d["eda"], t0=t[0], alpha=alpha)
            idx = ((r["ev_t"] - t[0]) * cfg.DFS).astype(int)
            row = {"pid": pid, "group": group, "alpha": alpha, "r2": r["r2"]}
            for block, mask in masks.items():
                sel = np.isin(idx, np.flatnonzero(mask))
                row[f"{block}_per_min"] = sel.sum() / (mask.sum() / cfg.DFS / 60)
                row[f"{block}_median_uS"] = np.median(r["ev_a"][sel]) if sel.any() else np.nan
            rows.append(row)
        print(f"P{pid} {group} done")

    v = pd.DataFrame(rows)
    out = cfg.ROOT / "data" / "cvxeda_real_alpha.csv"
    v.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 3):
        print(f"\n{v['pid'].nunique()} participants -> {out.name}, median per alpha:")
        print(v.groupby("alpha").median(numeric_only=True).drop(columns="pid"))


def cmd_window(pid=None, win_s=120):
    """Walk one short window through every stage, printing what each produces."""
    pid = pid or choose_participants()[0][0]
    d = dec.run(pid, load)

    film = np.flatnonzero(film_mask(d["marker"]))
    i0 = film[0] if len(film) else 0
    i1 = min(i0 + int(win_s * cfg.DFS), len(d["phasic"]))
    sl = slice(i0, i1)
    t = d["t"][sl]

    ev_t, ev_a = dec.extract_events(d["u"][sl], t0=t[0])
    d["win_ev_t"], d["win_ev_a"] = ev_t, ev_a

    r2 = 1 - (d["phasic"][sl] - d["fit"][sl]).var() / d["phasic"][sl].var()
    print(f"\nP{pid}, window {t[0]:.0f}-{t[-1]:.0f} s (block {d['marker'][i0]}), "
          f"{win_s} s at {cfg.DFS} Hz")
    print(f"  1 raw EDA        : {d['eda'][sl].min():.3f} to {d['eda'][sl].max():.3f} µS")
    print(f"  2 tonic          : {d['tonic'][sl].min():.3f} to {d['tonic'][sl].max():.3f} µS")
    print(f"  3 phasic         : {d['phasic'][sl].min():.3f} to {d['phasic'][sl].max():.3f} µS")
    print(f"  4 kernel         : tau_rise {d['tau_rise']:.2f} s, tau_decay {d['tau_decay']:.2f} s, "
          f"peak at {np.argmax(d['h']) / cfg.DFS:.1f} s")
    print(f"  5 impulses (u)   : {np.count_nonzero(d['u'][sl])} non-zero samples out of {i1 - i0}")
    print(f"  6 merged events  : {len(ev_t)} impulses, amplitudes "
          f"{ev_a.min():.3f}-{ev_a.max():.3f} µS, {len(ev_t) / (win_s / 60):.1f}/min")
    print(f"  7 check          : fit R2 = {r2:.3f}")

    figures.deconvolution_window(pid, d, sl, win_s)


def cmd_state(pid=99):
    """Deconvolution followed by state estimation, with the arousal figure."""
    d = dec.run(pid, load)
    s = st.run(d)
    t, mk, tb, xs = d["t"], d["marker"], s["t_bins"], s["state"]
    par = s["params"]

    print(f"P{pid}: {len(d['ev_t'])} impulses -> {len(xs)} bins of {cfg.BIN_S:.0f}s")
    print(f"  model: b0 {par['b0']:.2f}  g0 {par['g0']:.2f}  g1 {par['g1']:.2f}  "
          f"sr {par['sr']:.3f}  sv {par['sv']:.4f}")

    # which bins fall in the baseline and which in a film
    base = np.interp(tb, t, baseline_mask(mk).astype(float)) > .5
    film = np.interp(tb, t, film_mask(mk).astype(float)) > .5
    print(f"  mean arousal: baseline {xs[base].mean():+.3f}   films {xs[film].mean():+.3f}   "
          f"difference {xs[film].mean() - xs[base].mean():+.3f}")

    figures.arousal_state(pid, d, s)


COMMANDS = {
    "inspect": lambda pid: cmd_inspect(),
    "quality": lambda pid: run_screen(),
    "plot": lambda pid: cmd_plot(),
    "deconv": cmd_deconv,
    "window": cmd_window,
    "state": lambda pid: cmd_state(pid or 99),
    "sweep": experiments.lambda_sweep,
    "sweep-all": lambda pid: experiments.lambda_across_subjects(),
    "kernels": lambda pid: experiments.kernel_identifiability(),
    "tonic": lambda pid: experiments.tonic_window_sweep(),
    "synth": lambda pid: validation.run(),
    "synth-lam": lambda pid: validation.lambda_sweep(),
    "synth-fix": lambda pid: validation.compare_fixes(),
    "synth-robust": lambda pid: validation.robustness(),
    "synth-gcv": lambda pid: validation.gcv_test(),
    "synth-scale": lambda pid: validation.scale_test(),
    "cvx-smoke": lambda pid: validation.cvx_test(smoke=True),
    "cvx-synth": lambda pid: validation.cvx_test(),
    "cvx": lambda pid: cmd_cvx(pid or 99),
    "cvx-params": lambda pid: validation.cvx_params_test(),
    "cvx-real": lambda pid: cmd_cvx_real(),
    "cvx-alpha": lambda pid: validation.cvx_alpha_test(),
    "cvx-alpha-real": lambda pid: cmd_cvx_alpha_real(),
    "cvx-tau": lambda pid: validation.cvx_tau_test(),
    "tbme-export": tbme_export.export_all,
    "profile": lambda pid: cohort_profile.run(None if pid is None else [pid]),
    "synth-tbme": lambda pid: tbme_synth.run(),
    "score-tbme": lambda pid: tbme_score.run(),
    "synth-shift": lambda pid: tbme_synth.shift_inputs(),
    "score-shift": lambda pid: tbme_score.shift_test(),
    "false-events": lambda pid: tbme_score.false_events(),
    "cutoff-test": lambda pid: tbme_score.cutoff_test(),
    "window-test": lambda pid: tbme_score.window_test(),
    "synth-rate": lambda pid: tbme_synth.rate_inputs(),
    "rate-test": lambda pid: tbme_score.rate_test(),
    "tail-test": lambda pid: tbme_score.tail_test(),
    "tonic-test": lambda pid: tbme_score.tonic_test(),
    "fixed-tau-test": lambda pid: tbme_score.fixed_tau_test(),
    "tau-measure": lambda pid: tau_measure.run(),
    "tau-shared": lambda pid: tau_shared.run(),
    "preprocess-view": lambda pid: preprocess_view.run(),
    "paper-check": lambda pid: preprocess_view.paper_input_check(),
    "raw-atlas": lambda pid: own_preprocess.atlas(),
    "preprocess-stages": lambda pid: own_preprocess.stages(),
    "own-check": lambda pid: own_preprocess.check_inputs(),
    "own-export": lambda pid: own_preprocess.export(),
    "own4-export": lambda pid: own_preprocess.export(suffix="own4"),
    "own-score": lambda pid: own_preprocess.score(),
    "paper-export": lambda pid: ([tbme_export.paper_export(p) for p in tbme_score.PAPER_REAL_IDS],
                                 tbme_synth.paper_inputs()),
    "paper-test": lambda pid: tbme_score.paper_test(),
    "tbme-real": lambda pid: tbme_real.run(None if pid is None else [pid]),
    "importance-test": lambda pid: tbme_score.importance_test(),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    pid = int(argv[1]) if len(argv) > 1 else None
    COMMANDS[argv[0]](pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
