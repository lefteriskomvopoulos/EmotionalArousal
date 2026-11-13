"""
The synthetic generator from round 6 on (frozen 2026-11-11): realistic sessions with a known truth,
each built from one of the 50 selected participants, nothing fitted across participants.

  flat_session   one 100 Hz session: the participant's timeline, slow tonic (tonic_floor), gaps
                 and quantisation; their counted responses reused per run, each film its own, at
                 random times (tbme_synth.build_blocks), with one physiological shape
                 (draw_shape), capped at their largest counted SCR; slow fluctuations matched to
                 their count of small SCRs, then flat noise matched to their roughness, both on
                 the solver's input
  check50        synth-check: generated against real, counted the same way (data/synth50_check.csv)
  export_tune50  tune50-export / tune50-preview: the 18 tuning sessions U<pid> (round 6 on)
  export_test50  test50-export: the final test set V<pid>, one per participant, its own seed

Design, checks and every rejected version: docs/EXPERIMENTS.md Part 8 and
results/figures/tune50_preview/INSPECTION.md.  Seeds: CHECK_SEED, TUNE50_SEED, TEST6_SEED, each
+ the participant's id (+ 100000 / 200000 / 300000 for the other noise streams).

Saved: data/synth50_check.csv, data/noise_spectrum50_curves.csv, data/tune50_cohort.csv,
       data/test50_cohort.csv, data/tbme_input/<U, V><id>_own4(s).mat, data/tbme_synth/<U, V>*.npz,
       results/figures/tune50_vs_real.pdf, tune50_preview_vs_real.pdf
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import cohort_profile as cp
from . import config as cfg
from . import tbme_export as tx
from . import tbme_synth as ts
from .noise import SPECTRUM50_CURVES_CSV, band_power, colored_noise, matched, slow_deficit, spectra
from .preprocess import VARIANTS, low_pass, noise_2hz, prepare, typical_size

CHECK50_CSV = cfg.ROOT / "data" / "synth50_check.csv"
CHECK_SEED = 7070                        # apart from TEST_SEED and TUNE50_SEED
CHECK_BAND_HZ = (0.06, 0.3)
MIN_BLOCK_PEAKS = 8          # fewer counted SCRs in a block: its sizes are resampled from all of them
MIN_RUN_S = 60               # a run (one film, the baseline, a stretch between) shorter than this
                             # takes its block kind's rate: too short for a rate of its own
# Response shape: one per session, drawn at random within physiological limits (typical SCR rise
# 1-3 s onset to peak, half-recovery 2-10 s; Boucsein, Dawson et al.; own_config).  Not taken from
# the participant or from our solver: the test asks whether the solver recovers a shape it does
# not know, over the physiological range.  Within-person variation of shape is reported in the
# literature (Benedek & Kaernbach 2010) but not quantified, so none is added (a stated limit).
TP_RANGE_S = (1.0, 3.0)      # onset to peak
HALF_RANGE_S = (2.0, 10.0)   # peak to half recovery
MIN_DECAY_RATIO = 2.0        # decay time constant at least this many times the rise's
TAU_BOUNDS = ((0.1, 2.0), (1.5, 14.0))   # rise, decay: the solver's bounds (own_config)
FLUCT_BAND_HZ = (0.1, 0.5)
FLUCT_GRID_uS = np.r_[0.0, np.geomspace(0.002, 0.04, 10)]
TONIC_WIN_S = 60             # the tonic's window: the profile's floor (30 s) followed the recovery of
                             # large responses, which its smoothing turned into 30 s ramps with no
                             # response at their start (P118, P12, P173); tonic level changes over
                             # tens of seconds to minutes, fast rises are responses
SHAPE_CLASSES = ("fast", "typical", "slow")
TUNE50_CSV = cfg.ROOT / "data" / "tune50_cohort.csv"
TUNE50_SEED = 6060                       # apart from TEST_SEED and CHECK_SEED
TUNE50_QUOTA = {"neutral": 6, "positive": 4, "negative": 4, "mixed": 4}
TEST6_SEED = 8080                        # the final test set: apart from every other seed
TEST6_CSV = cfg.ROOT / "data" / "test50_cohort.csv"


def selected_ids():
    from . import selection
    sel = pd.read_csv(selection.SELECTION_CSV)
    return sorted(sel.loc[sel.selected.astype(bool), "id"].astype(int))


def gap_shape(times, t, mk):
    """Coefficient of variation of the gaps between consecutive SCRs inside one run (code_runs),
    minus the 1 s minimum: 1 is random timing, above 1 clustered.  A check only; NaN under 5 gaps."""
    runs = ts.code_runs(mk)
    starts = np.array([t[a] for a, _, _ in runs])
    times = np.sort(np.asarray(times, float))
    rid = np.searchsorted(starts, times, side="right") - 1
    d = np.clip(np.diff(times)[np.diff(rid) == 0] - cp.PEAK_GAP_S, 0, None)
    return float(d.std() / d.mean()) if len(d) >= 5 and d.mean() > 0 else np.nan


def counted_profile(times, amps, t, mk):
    """What counting (cohort_profile.scr_peaks) sees in one signal, a participant's peak table or a
    generated session alike.  Per run (ts.code_runs: each film, the baseline, each stretch
    between): rate and the log sizes counted there (the pool a session reuses), or its block
    kind's when shorter than MIN_RUN_S.  Per block kind: rate, log sizes, the median of those >=
    ts.SPLIT_uS.  Overall, for the checks: rate, rate >= and < ts.SPLIT_uS, median, p90, clustering."""
    times, amps = np.asarray(times, float), np.asarray(amps, float)
    lab = ts.block_labels(mk)
    idx = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    kinds = lab[idx]
    minutes = {b: (lab == b).sum() / cfg.FS / 60 for b in ts.BLOCKS}
    total = len(t) / cfg.FS / 60
    la, big = np.log(amps), amps >= ts.SPLIT_uS
    pool = {b: la[kinds == b] if (kinds == b).sum() >= MIN_BLOCK_PEAKS else la for b in ts.BLOCKS}
    rate = {b: (kinds == b).sum() / minutes[b] if minutes[b] > 0 else 0.0 for b in ts.BLOCKS}
    runs = ts.code_runs(mk)
    own = [(b - a) / cfg.FS >= MIN_RUN_S for a, b, _ in runs]
    rate_run = [((idx >= a) & (idx < b)).sum() / ((b - a) / cfg.FS / 60) if o else rate[k]
                for (a, b, k), o in zip(runs, own)]
    pool_run = [la[(idx >= a) & (idx < b)] if o and ((idx >= a) & (idx < b)).any() else pool[k]
                for (a, b, k), o in zip(runs, own)]
    return {"rate_pm": rate, "rate_run": rate_run, "pool_run": pool_run,
            "pool": pool,
            "med05": {b: float(np.median(amps[big & (kinds == b)])) if (big & (kinds == b)).any() else np.nan
                      for b in ts.BLOCKS},
            "scr_pm": len(amps) / total, "rate05": big.sum() / total, "rate_small": (~big).sum() / total,
            "median": float(np.median(amps)) if len(amps) else np.nan,
            "p90": float(np.percentile(amps, 90)) if len(amps) else np.nan,
            "cv": gap_shape(times, t, mk)}


def real_profiles(ids):
    """counted_profile of each participant's real peak table (cohort_profile, the 2 Hz recording)."""
    peaks = pd.read_csv(cp.PEAKS_CSV, dtype={"marker": str})
    out = {}
    for pid in ids:
        p = peaks[peaks.id == pid]
        t, mk = ts.timeline(pid)
        out[pid] = counted_profile(p.time_s.to_numpy(), p.amp_uS.to_numpy(), t, mk)
    return out


def tonic_floor(pid):
    """The tonic a synthetic session sits on, at the plain 2 Hz grid: the participant's recording
    (tbme_input/P<pid>.mat), its 5th percentile over TONIC_WIN_S, smoothed over TONIC_WIN_S."""
    from scipy.ndimage import percentile_filter, uniform_filter1d
    y2 = loadmat(tx.OUT / f"P{pid}.mat")["y"].ravel()
    w = int(TONIC_WIN_S * tx.FSY)
    return uniform_filter1d(percentile_filter(y2, cp.FLOOR_PCT, size=w, mode="nearest"), w, mode="nearest")


def fluctuation(n, rng, band=FLUCT_BAND_HZ):
    """Unit-SD Gaussian noise at cfg.FS with power only in band: the small slow wiggles real
    recordings carry everywhere (breathing, movement, small responses riding on others), which
    counting reads as 0.02-0.05 uS SCRs."""
    f = np.fft.rfftfreq(n, 1 / cfg.FS)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[(f < band[0]) | (f > band[1])] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def small_rate(y, fsy):
    """Counted 0.02-0.05 uS SCRs per minute on a solver input (cohort_profile.scr_peaks at its rate)."""
    return float((cp.scr_peaks(y, fsy)[1] < ts.SPLIT_uS).sum() / (len(y) / fsy / 60))


def flat_session(row, real, tr, td, seed, target, curves, caps, suffix="own4"):
    """One synthetic 100 Hz session as every session from round 6 on is made: the participant's
    timeline, slow tonic (tonic_floor), gaps and quantisation; their counted responses reused per run (each film its own) at
    random times (tbme_synth.build_blocks), shape tr / td, capped at their largest counted SCR;
    slow fluctuations (fluctuation) at the lowest level of FLUCT_GRID_uS that gives as many counted
    0.02-0.05 uS SCRs on the solver input as the participant's own input has (the closest level
    if none does); then flat noise matched to their roughness on the solver input (target, uS).
    Both are matched on the solver input, what the solver sees: counted on the plain 2 Hz signal,
    real small "responses" are partly fast ripples the 1 Hz filter removes, and matching there gave
    the tuning sessions a typical size on the solver input of 0.067 uS against the real 0.094.
    The two noise levels are the participant's own; nothing is set across participants.
    seed = (pulses, noise, fluctuations).  -> (t, mk, s, x_flat), s["fluct_uS"] the level used."""
    cutoff, fsy = VARIANTS[suffix]
    pid = int(row.id)
    t, mk = ts.timeline(pid)
    s = ts.build_blocks(t, mk, tonic_floor(pid), curves[f"gaps_P{pid}"], tr, td, real,
                        row.quant_uS if row.quant_uS > 0 else 0.01, seed[0], caps.get(pid, np.inf))
    fl = fluctuation(len(t), np.random.default_rng(seed[2]))
    want = small_rate(loadmat(tx.OUT / f"P{pid}_{suffix}.mat")["y"].ravel(), fsy)
    best = None
    for a in FLUCT_GRID_uS:
        x = matched(f"C{pid}", row, {**s, "x": s["x"] + a * fl}, t, {pid: target}, cutoff, fsy, seed=seed[1])
        got = small_rate(prepare(x, t, cfg.FS, (0, len(t) - 1), cutoff, fsy)["y"], fsy)
        err = abs(np.log(max(got, 1e-6) / want)) if want > 0 else got
        if best is None or err < best[0]:
            best = (err, a, x)
        if want <= 0 or got >= want:
            break
    s["fluct_uS"] = float(best[1])
    return t, mk, s, best[2]


def half_recovery(tau_rise, tau_decay):
    """Peak to half recovery of the response kernel, s."""
    k = ts.kernel(tau_rise, tau_decay, seconds=max(ts.KERNEL_S, 8 * tau_decay))
    ip = int(np.argmax(k))
    below = np.flatnonzero(k[ip:] <= 0.5)
    return below[0] / cfg.FS if len(below) else np.inf


def draw_shape(rng, want=None):
    """One session's (tau_rise, tau_decay), uniform over the solver's bounds (TAU_BOUNDS) and kept
    only when physiological: peak TP_RANGE_S after onset, half-recovery HALF_RANGE_S, decay at least
    MIN_DECAY_RATIO x the rise; and, if want is given, only in that class.  -> (tau_rise,
    tau_decay, label by half-recovery: fast < 4 s, typical 4-7 s, slow > 7 s)."""
    while True:
        tr, td = rng.uniform(*TAU_BOUNDS[0]), rng.uniform(*TAU_BOUNDS[1])
        if td < MIN_DECAY_RATIO * tr or not TP_RANGE_S[0] <= ts.tp_of(tr, td) <= TP_RANGE_S[1]:
            continue
        h = half_recovery(tr, td)
        if HALF_RANGE_S[0] <= h <= HALF_RANGE_S[1]:
            label = "fast" if h < 4 else ("slow" if h > 7 else "typical")
            if want is None or label == want:
                return float(tr), float(td), label


def _ratio(a, b):
    return a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def check50(suffix="own4"):
    """synth-check: how well the round-6 generator fits the 50 selected participants, and the
    slow noise its flat sessions lack.

    The generator (flat_session), with nothing set across participants: each session copies its
    participant's timeline, slow tonic (60 s), gaps and quantisation, reuses their counted responses
    per run (each film, the baseline, each stretch between) at random times, gives them one
    physiological shape drawn at random (draw_shape), gets slow fluctuations at the level that
    gives their count of 0.02-0.05 uS SCRs (a first version without them counted 44% of the
    real small ones) and noise matched to their roughness on the solver input.  One session per
    participant here, its own seeds (CHECK_SEED); these
    sessions serve nothing else.  Counted the same way as the real recording, generated / real:
    rate, rate of SCRs >= and < 0.05 uS, median and p90 size, median size >= 0.05 uS in baseline
    and films, rate per block, clustering, roughness.  The known gap: overlapping real responses
    count as one, so a generated session holds slightly fewer, larger responses than its
    participant really had, and some counted small ones that were noise become true pulses.
    Then the slow noise (the slow-noise versions): real minus synthetic quiet-stretch spectrum of
    each participant's input against their flat session, median over the pairs.

    Earlier tries (docs/EXPERIMENTS.md, results/figures/tune50_preview/INSPECTION.md) added cohort
    correction factors to match the counted numbers; they could not all be matched (real
    recordings hold more small fluctuations than pulses and noise make), and forcing them drove
    the generator to implausible settings.  Nothing is fitted to these 50 beyond their own counts.
    """
    cutoff, fsy = VARIANTS[suffix]
    ids = selected_ids()
    prof = pd.read_csv(cp.PROFILE_CSV)
    curves = np.load(cp.CURVES_NPZ)
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    rp = real_profiles(ids)
    rng = np.random.default_rng(CHECK_SEED)
    rows, spec = [], []
    for k, row in enumerate(prof[prof.id.isin(ids)].sort_values("scr_pm").itertuples(), 1):
        pid = int(row.id)
        tr, td, label = draw_shape(rng)
        rm = loadmat(tx.OUT / f"P{pid}_{suffix}.mat")
        real, real_mk = rm["y"].ravel(), rm["marker"].ravel()
        t, mk, s, x = flat_session(row, rp[pid], tr, td, (CHECK_SEED + pid, CHECK_SEED + 200000 + pid, CHECK_SEED + 300000 + pid),
                                   noise_2hz(real), curves, caps, suffix)
        _, ex = cp.measure(x, t, mk)
        gp, r0 = counted_profile(ex["peaks_t"], ex["peaks_amp"], t, mk), rp[pid]
        p = prepare(x, t, cfg.FS, (0, len(t) - 1), cutoff, fsy)
        syn = p["y"]
        syn_mk = pd.to_numeric(pd.Series(mk[p["idx"]]), errors="coerce").to_numpy(float)
        (sr, nr), (ss, ns) = spectra(real, fsy), spectra(syn, fsy)
        (fr, pr), (_, ps) = sr["quiet"], ss["quiet"]
        spec += [{"source_id": pid, "kind": "quiet", "f_hz": a, "psd_real": x_, "psd_synthetic": y_}
                 for a, x_, y_ in zip(fr, pr, ps)]
        rows.append({"source_id": pid, "group": row.group, "shape": label, "tau_rise": tr, "tau_decay": td,
                     "tp_s": ts.tp_of(tr, td), "half_s": half_recovery(tr, td), "fluct_uS": s["fluct_uS"],
                     "rough": _ratio(noise_2hz(syn), noise_2hz(real)),
                     "small_in": _ratio(small_rate(syn, fsy), small_rate(real, fsy)),
                     "scale_in": _ratio(typical_size(syn, syn_mk)["scale_uS"],
                                        typical_size(real, real_mk)["scale_uS"]),
                     "rate": _ratio(gp["scr_pm"], r0["scr_pm"]),
                     "rate05": _ratio(gp["rate05"], r0["rate05"]),
                     "small": _ratio(gp["rate_small"], r0["rate_small"]),
                     "size": _ratio(gp["median"], r0["median"]),
                     "p90": _ratio(gp["p90"], r0["p90"]),
                     "base_size": _ratio(gp["med05"]["baseline"], r0["med05"]["baseline"]),
                     "film_size": _ratio(gp["med05"]["film"], r0["med05"]["film"]),
                     "base_rate": _ratio(gp["rate_pm"]["baseline"], r0["rate_pm"]["baseline"]),
                     "film_rate": _ratio(gp["rate_pm"]["film"], r0["rate_pm"]["film"]),
                     "gap_cv": _ratio(gp["cv"], r0["cv"]),
                     "quiet_windows_real": nr, "quiet_windows_syn": ns})
        if k % 10 == 0:
            print(f"  {k}/{len(ids)} participants")
    res = pd.DataFrame(rows)
    res.to_csv(CHECK50_CSV, index=False)
    pd.DataFrame(spec).groupby(["kind", "f_hz"])[["psd_real", "psd_synthetic"]].median().reset_index() \
        .to_csv(SPECTRUM50_CURVES_CSV, index=False)

    cols = ["rough", "small_in", "scale_in", "rate", "rate05", "small", "size", "p90", "base_size",
            "film_size", "base_rate", "film_rate", "gap_cv"]
    t = res.groupby("shape")[cols].median()
    t.loc["all"] = res[cols].median()
    print(f"\nshapes drawn: peak after {res.tp_s.min():.2f}-{res.tp_s.max():.2f} s, half-recovery"
          f" {res.half_s.min():.1f}-{res.half_s.max():.1f} s (median {res.half_s.median():.1f});"
          f" {', '.join(f'{k} {v}' for k, v in res['shape'].value_counts().items())}")
    top = int((res.fluct_uS >= FLUCT_GRID_uS[-1]).sum())
    print(f"slow fluctuations (SD, uS): median {res.fluct_uS.median():.4f}, {res.fluct_uS.min():.4f}-"
          f"{res.fluct_uS.max():.4f}; none needed for {int((res.fluct_uS == 0).sum())}, grid top for {top}")
    print("generated / real participant, same counting (1 = as real), median by shape:")
    print(t.to_string(float_format=lambda v: f"{v:.2f}"))
    print("  all, 10-90%:", ", ".join(f"{c} {res[c].quantile(.1):.2f}-{res[c].quantile(.9):.2f}" for c in cols[:8]))
    print("  (rate, size, p90: all counted SCRs >= 0.02 uS; rate05 and the block sizes: >= 0.05 uS;"
          " small: 0.02-0.05 uS; gap_cv: clustering; small_in, scale_in: on the solver input, what it sees)")
    both = int(((res.quiet_windows_real > 0) & (res.quiet_windows_syn > 0)).sum())
    fq, d = slow_deficit(SPECTRUM50_CURVES_CSV)
    print(f"slow noise from {both} pairs with quiet stretches in both: power the flat sessions lack,"
          f" {CHECK_BAND_HZ[0]}-{CHECK_BAND_HZ[1]} Hz, {band_power(fq, d, *CHECK_BAND_HZ):.2e} uS^2"
          f" (first estimate {band_power(*slow_deficit(), *CHECK_BAND_HZ):.2e})")
    print("saved", CHECK50_CSV.name, SPECTRUM50_CURVES_CSV.name)
    return res



def synth_pair(row, g, tr, td, names, seeds, suffix, deficit, curves, caps, real, target, save=True):
    """Two synthetic sessions from one participant (flat_session), saved with their truth:
    names[0] with flat noise matched to the participant's roughness (target, uS), names[1] the
    same plus slow noise (deficit, from slow_deficit).  real: the participant's counted_profile.
    seeds = (pulses, flat noise, slow noise, fluctuations).
    -> the session's numbers, with what counting sees in the generated flat signal next to the
    participant's, and under "_plot" the flat input and its truth for a figure.
    save=False writes nothing."""
    cutoff, fsy = VARIANTS[suffix]
    pid = int(row.id)
    t, mk, s, x_flat = flat_session(row, real, tr, td, (seeds[0], seeds[1], seeds[3]), target, curves, caps, suffix)
    x_slow = x_flat + colored_noise(len(t), cfg.FS, *deficit, np.random.default_rng(seeds[2]))
    _, ex = cp.measure(x_flat, t, mk)
    gp = counted_profile(ex["peaks_t"], ex["peaks_amp"], t, mk)
    out = {"tau_group": g, "tau_rise": tr, "tau_decay": td, "tp_s": ts.tp_of(tr, td), "half_s": half_recovery(tr, td),
           "n_true": len(s["ev_t"]),
           "amp_med_true_uS": float(np.median(s["ev_a"])) if len(s["ev_a"]) else np.nan,
           "real_scr_pm": real["scr_pm"], "syn_scr_pm": gp["scr_pm"],
           "real_scr_amp_med_uS": real["median"], "syn_scr_amp_med_uS": gp["median"],
           "real_scr_amp_p90_uS": real["p90"], "syn_scr_amp_p90_uS": gp["p90"],
           "real_gap_cv": real["cv"], "syn_gap_cv": gp["cv"],
           **{f"{w}_med_{b}_uS": p["med05"][b] for w, p in (("real", real), ("syn", gp)) for b in ("baseline", "film")},
           "real_rate05_pm": real["rate05"], "syn_rate05_pm": gp["rate05"],
           "real_small_pm": real["rate_small"], "syn_small_pm": gp["rate_small"],
           "rough_real_uS": target, "fluct_uS": s["fluct_uS"]}
    for kind, name, x in (("flat", names[0], x_flat), ("slow", names[1], x_slow)):
        p = prepare(x, t, cfg.FS, (0, len(t) - 1), cutoff, fsy)
        if save:
            tx.save(name, p["y"], p["t2"], mk[p["idx"]], int(p["gaps"].sum()), fs=fsy)
            np.savez_compressed(ts.TRUTH_DIR / f"{name}.npz", ev_t=s["ev_t"], ev_a=s["ev_a"],
                                tonic2=low_pass(s["tonic"], cfg.FS, cutoff)[p["idx"]] + p["shift"],
                                phasic2=low_pass(s["phasic"], cfg.FS, cutoff)[p["idx"]],
                                tau_rise=tr, tau_decay=td, t2=p["t2"], source_id=pid)
        out[f"rough_{kind}_uS"] = noise_2hz(p["y"])
        if kind == "flat":
            out["_plot"] = {"t_syn": p["t2"], "y_syn": p["y"],
                            "mk_syn": pd.to_numeric(pd.Series(mk[p["idx"]]), errors="coerce").to_numpy(float),
                            "ev_t": s["ev_t"], "ev_a": s["ev_a"]}
    return out


def export_tune50(suffix="own4", save=True):
    """tune50-export: the tuning sessions of round 6 on, built from the 50 selected participants
    (selection.py), so the settings are tuned on the population they are used on.
    save=False (tune50-preview): the same sessions, same draw and seeds, only plotted
    (results/figures/tune50_preview*); no input, truth or cohort file is written.

    18 of the 50 by a seeded draw within each film group (TUNE50_QUOTA, in proportion to 20/10/10/10),
    each with one physiological response shape drawn at random (draw_shape), stratified so the
    18 cover the range evenly: sorted by SCR rate, dealt in turn to fast / typical / slow
    half-recovery (6 each; an unstratified draw gave 1 / 11 / 6).  Each gives U<pid>_<suffix> (flat noise) and U<pid>_<suffix>s
    (plus slow noise), with their own seeds, so no pulse or noise is shared with a test session.
    Generator: flat_session.  Slow noise from synth-check, which needs final-export (the real
    inputs P<pid>_<suffix>, whose roughness the noise is matched to)."""
    from . import selection
    sel = pd.read_csv(selection.SELECTION_CSV)
    sel = sel[sel.selected.astype(bool)]
    rng = np.random.default_rng(TUNE50_SEED)
    ids = []
    for group, n in TUNE50_QUOTA.items():
        ids += [int(v) for v in rng.choice(np.sort(sel.loc[sel.group == group, "id"].astype(int).to_numpy()),
                                            n, replace=False)]
    prof = pd.read_csv(cp.PROFILE_CSV)
    picked = prof[prof.id.isin(ids)].sort_values("scr_pm")
    deficit = slow_deficit(SPECTRUM50_CURVES_CSV)
    curves = np.load(cp.CURVES_NPZ)
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    rp = real_profiles(ids)
    rows, plots = [], {}
    for k, row in enumerate(picked.itertuples()):
        pid = int(row.id)
        tr, td, g = draw_shape(rng, SHAPE_CLASSES[k % 3])
        target = noise_2hz(loadmat(tx.OUT / f"P{pid}_{suffix}.mat")["y"].ravel())
        seeds = (TUNE50_SEED + pid, TUNE50_SEED + 200000 + pid, TUNE50_SEED + 100000 + pid, TUNE50_SEED + 300000 + pid)
        out = synth_pair(row, g, tr, td, (f"U{pid}_{suffix}", f"U{pid}_{suffix}s"), seeds, suffix,
                         deficit, curves, caps, rp[pid], target, save=save)
        plots[f"U{pid}"] = out.pop("_plot")
        rows.append({"session": f"U{pid}", "source_id": pid, "group": row.group, **out})
    c = pd.DataFrame(rows)
    checks = {"rate": ("syn_scr_pm", "real_scr_pm"), "size": ("syn_scr_amp_med_uS", "real_scr_amp_med_uS"),
              "p90": ("syn_scr_amp_p90_uS", "real_scr_amp_p90_uS"), "rate05": ("syn_rate05_pm", "real_rate05_pm"),
              "small": ("syn_small_pm", "real_small_pm"), "gap_cv": ("syn_gap_cv", "real_gap_cv"),
              "base_size": ("syn_med_baseline_uS", "real_med_baseline_uS"),
              "film_size": ("syn_med_film_uS", "real_med_film_uS")}
    for k, (a, b) in checks.items():
        c[f"{k}_ratio"] = c[a] / c[b]
    if save:
        c.to_csv(TUNE50_CSV, index=False)
    print("sessions per film group and response shape (half-recovery: fast < 4 s, typical 4-7 s, slow > 7 s):")
    print(pd.crosstab(c.group, c.tau_group, margins=True).to_string())
    print("\ngenerated / real participant, same counting, median by shape (rate, size, p90: SCRs >= 0.02 uS;"
          " rate05, gap_cv and the block sizes: >= 0.05 uS):")
    cols = [f"{k}_ratio" for k in checks]
    t = c.groupby("tau_group")[cols].median()
    t.loc["all"] = c[cols].median()
    t.columns = list(checks)
    print(t.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"response peak after onset: {c.tp_s.min():.2f}-{c.tp_s.max():.2f} s; slow fluctuations (SD)"
          f" {c.fluct_uS.min():.4f}-{c.fluct_uS.max():.4f} uS, median {c.fluct_uS.median():.4f}")
    print(f"roughness, median: real {c.rough_real_uS.median():.4f}, flat {c.rough_flat_uS.median():.4f},"
          f" slow {c.rough_slow_uS.median():.4f} uS")
    if not save:
        _plot_sessions(c, suffix, "tune50_preview", "tune50_preview_vs_real.pdf", plots)
        print("preview only: no input, truth or cohort file was written")
        return c
    print(f"saved {len(c)} flat (U*_{suffix}) and {len(c)} slow-noise (U*_{suffix}s) inputs in {tx.OUT.name},"
          f" truth in {ts.TRUTH_DIR.name}; {TUNE50_CSV.name}")
    return c


def export_test50(suffix="own4"):
    """test50-export: the final test set, one session per selected participant (V<pid>), by the
    frozen round-6 generator (flat_session, synth_pair) with its own seed, so nothing is shared
    with the tuning or check sessions.  Shapes drawn unstratified (draw_shape), as they come.
    Unseen by tuning; scored once by test50-score.  Needs final-export and synth-check."""
    ids = selected_ids()
    prof = pd.read_csv(cp.PROFILE_CSV)
    picked = prof[prof.id.isin(ids)].sort_values("scr_pm")
    deficit = slow_deficit(SPECTRUM50_CURVES_CSV)
    curves = np.load(cp.CURVES_NPZ)
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    rp = real_profiles(ids)
    rng = np.random.default_rng(TEST6_SEED)
    rows = []
    for k, row in enumerate(picked.itertuples(), 1):
        pid = int(row.id)
        tr, td, g = draw_shape(rng)
        target = noise_2hz(loadmat(tx.OUT / f"P{pid}_{suffix}.mat")["y"].ravel())
        seeds = (TEST6_SEED + pid, TEST6_SEED + 200000 + pid, TEST6_SEED + 100000 + pid, TEST6_SEED + 300000 + pid)
        out = synth_pair(row, g, tr, td, (f"V{pid}_{suffix}", f"V{pid}_{suffix}s"), seeds, suffix,
                         deficit, curves, caps, rp[pid], target)
        out.pop("_plot")
        rows.append({"session": f"V{pid}", "source_id": pid, "group": row.group, **out})
        if k % 10 == 0:
            print(f"  {k}/{len(ids)} participants")
    c = pd.DataFrame(rows)
    c.to_csv(TEST6_CSV, index=False)
    print("sessions per film group and response shape:")
    print(pd.crosstab(c.group, c.tau_group, margins=True).to_string())
    print(f"half-recovery {c.half_s.min():.1f}-{c.half_s.max():.1f} s, peak {c.tp_s.min():.2f}-"
          f"{c.tp_s.max():.2f} s; slow fluctuations median {c.fluct_uS.median():.4f} uS;"
          f" roughness real {c.rough_real_uS.median():.4f}, generated {c.rough_flat_uS.median():.4f} uS")
    print(f"saved {len(c)} test inputs V*_{suffix} in {tx.OUT.name}, truth in {ts.TRUTH_DIR.name}; {TEST6_CSV.name}")
    return c


def plot_tune50(suffix="own4"):
    """tune50-plots: each tuning session (U, flat noise) next to the real participant it was built
    from, both as the solver gets them; true pulses marked.  Ordered by tau group, then film group.
    -> results/figures/tune50_vs_real.pdf, results/figures/tune50/*.png"""
    _plot_sessions(pd.read_csv(TUNE50_CSV), suffix, "tune50", "tune50_vs_real.pdf")


def _plot_sessions(c, suffix, folder, pdf_name, syn=None):
    """The figures of plot_tune50, from the saved sessions, or from syn {session: arrays} (preview)."""
    from . import figures
    items = []
    for r in c.sort_values(["tau_group", "group", "source_id"]).itertuples():
        real = loadmat(tx.OUT / f"P{r.source_id}_{suffix}.mat")
        if syn is None:
            m = loadmat(tx.OUT / f"{r.session}_{suffix}.mat")
            tr = np.load(ts.TRUTH_DIR / f"{r.session}_{suffix}.npz")
            s = {"t_syn": m["t"].ravel(), "y_syn": m["y"].ravel(), "mk_syn": m["marker"].ravel(),
                 "ev_t": tr["ev_t"], "ev_a": tr["ev_a"]}
        else:
            s = syn[r.session]
        n_vis = int((s["ev_a"] >= cfg.MIN_EVENT_uS).sum())
        p90 = getattr(r, "real_scr_amp_p90_uS", np.nan), getattr(r, "syn_scr_amp_p90_uS", np.nan)
        g = lambda name: getattr(r, name, np.nan)
        blocks = (f"\nSCRs 0.02-0.05 uS: {g('real_small_pm'):.2f} / {g('syn_small_pm'):.2f} per min;"
                  f" >= 0.05 uS: {g('real_rate05_pm'):.2f} / {g('syn_rate05_pm'):.2f} per min, median size"
                  f" baseline {g('real_med_baseline_uS'):.3f} / {g('syn_med_baseline_uS'):.3f} uS,"
                  f" films {g('real_med_film_uS'):.3f} / {g('syn_med_film_uS'):.3f} uS, clustering (gap CV)"
                  f" {g('real_gap_cv'):.2f} / {g('syn_gap_cv'):.2f}; {r.tau_group} shape, peak"
                  f" {g('tp_s'):.2f} s after onset; slow fluctuations {g('fluct_uS'):.4f} uS")
        items.append({
            "name": r.session, "label": f"{r.session} / P{r.source_id}\n{r.group}, {r.tau_group}",
            "real_name": f"real P{r.source_id} ({r.group} films)",
            "syn_name": f"synthetic {r.session}, built from P{r.source_id}",
            "title": (f"{r.session} from P{r.source_id} ({r.group} films), tau group {r.tau_group}: true tau"
                      f" {r.tau_rise:.2f} s rise / {r.tau_decay:.2f} s decay, {r.n_true} true pulses,"
                      f" {n_vis} of them >= 0.02 uS\n"
                      f"counted SCRs real / synthetic: {r.real_scr_pm:.2f} / {r.syn_scr_pm:.2f} per min,"
                      f" median size {r.real_scr_amp_med_uS:.3f} / {r.syn_scr_amp_med_uS:.3f} uS,"
                      f" 90th percentile {p90[0]:.3f} / {p90[1]:.3f} uS;"
                      f" roughness of the input {r.rough_real_uS:.4f} / {r.rough_flat_uS:.4f} uS" + blocks),
            "t_real": real["t"].ravel(), "y_real": real["y"].ravel(), "mk_real": real["marker"].ravel(),
            **s})
    figures.synthetic_vs_real(items, folder, pdf_name)
