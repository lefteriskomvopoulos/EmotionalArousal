"""
The first synthetic generator and its runs: tuning rounds 1-5 and the first final run.
Superseded by generator.py from round 6 on (docs/EXPERIMENTS.md Parts 4, 5, 7); kept unchanged so
every number from those rounds can be reproduced.  The 36 sessions themselves are built by
tbme_synth.py (synth), each from one real participant's profile, with cohort correction factors.

The 36 synthetic sessions are split once into 18 tuning and 18 test sessions, 6 + 6 of each tau
group.  Only tuning sessions are shown: the test sessions stay unseen until settings are frozen.

  split            18 tuning + 18 test sessions of the 36, drawn once (data/own_split.csv)
  export           own-export / own4-export: the 36 (S<k>) and 5 real inputs, with truth
  export_selected  final-export: the 50 selected participants' real inputs P<pid>_own4 (still the
                   real inputs of every later run) and their first test sessions F / T
  score_final      final-score: the first final run on F / T
  noise_spectrum   noise-spectrum: the S sessions' noise against their sources, band by band

Saved: data/own_split.csv, data/input_check_<variant>.csv, data/tbme_test50_cohort.csv,
       data/own_final_test_scores.csv, data/noise_spectrum.csv, data/noise_spectrum_curves.csv,
       data/tbme_input/<S, F, T, P><id>_<variant>.mat, data/tbme_synth/<S, F, T><id>_<variant>.npz
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import cohort_profile as cp
from . import config as cfg
from . import tbme_export as tx
from . import tbme_synth as ts
from .dataset import load
from .noise import (BANDS_HZ, SPECTRUM_CSV, SPECTRUM_CURVES_CSV, band_power, colored_noise, matched,
                    slow_deficit, spectra)
from .preprocess import CHECK_CSV, RAW_FS, REAL_IDS, VARIANTS, low_pass, noise_2hz, prepare, segment_of
from .runs import FINAL_OUT

SPLIT_CSV = cfg.ROOT / "data" / "own_split.csv"
SPLIT_SEED = 2026
TEST50_COHORT_CSV = cfg.ROOT / "data" / "tbme_test50_cohort.csv"
TEST_SEED = 5050
FINAL_SETTING = "own4_p1000_k14_f3"
FINAL_TEST_CSV = cfg.ROOT / "data" / "own_final_test_scores.csv"


def split():
    """18 tuning + 18 test sessions, 6 + 6 per tau group.  Drawn once, then read back."""
    if SPLIT_CSV.exists():
        return pd.read_csv(SPLIT_CSV)
    cohort = pd.read_csv(ts.COHORT_CSV)
    rng = np.random.default_rng(SPLIT_SEED)
    rows = []
    for group, g in cohort.groupby("tau_group", sort=True):
        order = rng.permutation(g.session.to_numpy())
        rows += [{"session": s, "tau_group": group, "role": "tune" if k < len(order) // 2 else "test"}
                 for k, s in enumerate(order)]
    out = pd.DataFrame(rows).sort_values(["role", "tau_group", "session"], ascending=[False, True, True])
    out.to_csv(SPLIT_CSV, index=False)
    return out


def variant_targets(suffix):
    """Roughness of the synthetic sessions' source participants through a variant's chain."""
    cutoff, fsy = VARIANTS[suffix]
    ids = sorted(set(pd.read_csv(ts.COHORT_CSV).source_id.astype(int)))
    rows = []
    for pid in ids:
        d = load(pid, every=1)
        p = prepare(d["eda"], d["t"], RAW_FS, segment_of(d["marker"]), cutoff, fsy)
        rows.append({"id": pid, "noise_uS": noise_2hz(p["y"])})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.ROOT / "data" / f"input_check_{suffix}.csv", index=False)
    print(f"{suffix}: roughness of {len(out)} source participants, median {out.noise_uS.median():.4f} uS")
    return out.set_index("id").noise_uS.to_dict()


def tuning_sessions(n=None):
    """The rebuilt tuning sessions, sizes capped."""
    roles = split()
    tune = roles[roles.role == "tune"]
    wanted = tune.session if n is None else [
        s for group, k in n for s in tune[tune.tau_group == group].session.iloc[:k]]
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    return ts.rebuilt(set(wanted), caps)


def export(real_ids=REAL_IDS, suffix="own"):
    """The inputs, through prepare(): all 36 synthetic sessions with their truth, and real ones.

    suffix picks the variant (VARIANTS).  'own' matches noise to own-check's 119; any other
    variant first measures its own targets on the source participants (variant_targets).
    """
    cutoff, fsy = VARIANTS[suffix]
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    if suffix == "own":
        check = pd.read_csv(CHECK_CSV)
        targets = check.set_index("id").noise_2hz_uS.to_dict()
    else:
        targets = variant_targets(suffix)
    tune, rough = set(split().query("role == 'tune'").session), []
    for name, row, g, tr, td, t, mk, s in ts.rebuilt(None, caps):
        p = prepare(matched(name, row, s, t, targets, cutoff, fsy), t, cfg.FS, (0, len(t) - 1), cutoff, fsy)
        if name in tune:
            rough.append(noise_2hz(p["y"]))
        tx.save(f"{name}_{suffix}", p["y"], p["t2"], mk[p["idx"]], int(p["gaps"].sum()), fs=fsy)
        np.savez_compressed(ts.TRUTH_DIR / f"{name}_{suffix}.npz", ev_t=s["ev_t"], ev_a=s["ev_a"],
                            tonic2=low_pass(s["tonic"], cfg.FS, cutoff)[p["idx"]] + p["shift"],
                            phasic2=low_pass(s["phasic"], cfg.FS, cutoff)[p["idx"]],
                            tau_rise=tr, tau_decay=td, t2=p["t2"], source_id=int(row.id))
    for pid in real_ids:
        d = load(pid, every=1)
        p = prepare(d["eda"], d["t"], RAW_FS, segment_of(d["marker"]), cutoff, fsy)
        tx.save(f"P{pid}_{suffix}", p["y"], p["t2"], d["marker"][p["idx"]], int(p["gaps"].sum()), fs=fsy)
        print(f"P{pid}_{suffix}: {p['n_jumps']} jumps, shifted +{p['shift']:.3f} uS, {len(p['y'])} samples")
    print(f"median roughness ({cutoff} Hz, {fsy} Hz): real sources {np.median(list(targets.values())):.4f}, "
          f"18 tuning {np.median(rough):.4f} uS")
    print("saved 36 synthetic and", len(real_ids), "real inputs in", tx.OUT.name, "with suffix", suffix)


def export_selected(suffix="own4"):
    """final-export: the inputs of the final run, for the 50 selected participants (selection.py).

    For each: the real input P<pid>_<suffix>, and two synthetic test sessions built from the same
    participant's profile, with their truth.  Both use the tuning sessions' generator and
    correction factors (per tau group, from the cohort table), tau design (prior / moderate / far
    dealt by SCR rate, new seed) and flat noise matched to each participant's roughness:
      F<pid>  exactly that, as the tuning sessions
      T<pid>  the same signal plus the slow noise the tuning comparison suggested was missing:
              spectrum real minus synthetic of the quiet stretches (slow_deficit, 0.03-0.6 Hz,
              cohort median, absolute units)
    Same responses, same flat noise, so F vs T isolates the slow noise.  Why two: the real noise
    shape cannot be pinned.  Quiet-stretch power is not a clean noise measure: response tails
    leak into it, and with the same added noise the real / synthetic ratio still ran from
    0.03-0.3 (far, 5 s decays) to 0.35-1.7 (moderate).  So the final score is reported as the
    range F-T.  (A first T version shaped all added noise like the real spectrum and scaled it
    by roughness; roughness at 4 Hz is set mostly by 1-2 Hz, which the low-pass removes from
    added noise, so its slow part was inflated.)
    Unseen by tuning; scored once by final-score.
    Superseded for what we report: built with the first cohort's correction factors (its sessions
    measured about 9% fewer and 23% larger SCRs than the participants, docs/EXPERIMENTS.md) and the
    first slow-noise estimate.  Kept unchanged so that score can be reproduced.
    """
    from . import selection
    cutoff, fsy = VARIANTS[suffix]
    sel = pd.read_csv(selection.SELECTION_CSV)
    ids = sorted(sel.loc[sel.selected.astype(bool), "id"].astype(int))
    deficit = slow_deficit()
    prof = pd.read_csv(cp.PROFILE_CSV)
    picked = prof[prof.id.isin(ids)].sort_values("scr_pm")
    curves = np.load(cp.CURVES_NPZ)
    caps = pd.read_csv(cp.PEAKS_CSV).groupby("id").amp_uS.max().to_dict()
    fac = pd.read_csv(ts.COHORT_CSV).groupby("tau_group")[["c_rate", "c_amp", "c_noise"]].first()

    rows = []
    for k, (row, g, tr, td) in enumerate(ts.assign_tau(picked, np.random.default_rng(TEST_SEED)), 1):
        pid, name = int(row.id), f"T{int(row.id)}"
        d = load(pid, every=1)                                         # the real input
        pr = prepare(d["eda"], d["t"], RAW_FS, segment_of(d["marker"]), cutoff, fsy)
        tx.save(f"P{pid}_{suffix}", pr["y"], pr["t2"], d["marker"][pr["idx"]], int(pr["gaps"].sum()), fs=fsy)
        target = {pid: noise_2hz(pr["y"])}

        t, mk = ts.timeline(pid)                                       # the test session
        f = fac.loc[g]
        s = ts.build(row, t, mk, curves[f"floor_P{pid}"], curves[f"gaps_P{pid}"], tr, td,
                     f.c_rate, f.c_amp, f.c_noise, TEST_SEED + pid, amp_cap=caps.get(pid, np.inf))
        x_flat = matched(name, row, s, t, target, cutoff, fsy)
        x_slow = x_flat + colored_noise(len(t), cfg.FS, *deficit,
                                        np.random.default_rng(TEST_SEED + 100000 + pid))
        sr, nr = spectra(pr["y"], fsy)                                 # how real does each look?
        slow = lambda sp, nq: band_power(*sp["quiet"], 0.06, 0.3) if nq else np.nan
        out = {"session": name, "source_id": pid, "group": row.group, "tau_group": g,
               "tau_rise": tr, "tau_decay": td, "n_true": len(s["ev_t"]),
               "rate_true_pm": len(s["ev_t"]) / row.dur_min,
               "amp_med_true_uS": float(np.median(s["ev_a"])) if len(s["ev_a"]) else np.nan,
               "rough_real_uS": target[pid], "slow_real_uS2": slow(sr, nr)}
        for prefix, x in (("F", x_flat), ("T", x_slow)):
            p = prepare(x, t, cfg.FS, (0, len(t) - 1), cutoff, fsy)
            tx.save(f"{prefix}{pid}_{suffix}", p["y"], p["t2"], mk[p["idx"]], int(p["gaps"].sum()), fs=fsy)
            np.savez_compressed(ts.TRUTH_DIR / f"{prefix}{pid}_{suffix}.npz", ev_t=s["ev_t"], ev_a=s["ev_a"],
                                tonic2=low_pass(s["tonic"], cfg.FS, cutoff)[p["idx"]] + p["shift"],
                                phasic2=low_pass(s["phasic"], cfg.FS, cutoff)[p["idx"]],
                                tau_rise=tr, tau_decay=td, t2=p["t2"], source_id=pid)
            ss, ns = spectra(p["y"], fsy)
            out[f"rough_{prefix}_uS"] = noise_2hz(p["y"])
            out[f"slow_{prefix}_uS2"] = slow(ss, ns)
        rows.append(out)
        if k % 10 == 0:
            print(f"  {k}/{len(ids)} participants")
    c = pd.DataFrame(rows)
    for prefix in ("F", "T"):
        c[f"slow_ratio_real_{prefix}"] = c.slow_real_uS2 / c[f"slow_{prefix}_uS2"]
    c.to_csv(TEST50_COHORT_CSV, index=False)
    print(c.groupby("group").size().to_string())
    print(f"roughness, median: real {c.rough_real_uS.median():.4f}, flat F {c.rough_F_uS.median():.4f},"
          f" slow T {c.rough_T_uS.median():.4f} uS")
    print("quiet-stretch power 0.06-0.3 Hz, real / test session, median per tau group"
          " (not a clean noise measure: response tails leak in)")
    print(c.groupby("tau_group")[["slow_ratio_real_F", "slow_ratio_real_T"]].median()
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"saved {len(c)} real (P*), {len(c)} flat (F*) and {len(c)} slow-noise (T*) inputs, suffix"
          f" {suffix}, in {tx.OUT.name}; truth in {ts.TRUTH_DIR.name}; {TEST50_COHORT_CSV.name}")
    return c


def score_final(root=FINAL_OUT / FINAL_SETTING):
    """final-score: the frozen setting on the 50 test participants, both noise versions (F flat,
    T with slow noise), once, against their truth.  The reported score is the range F-T."""
    from . import tbme_score as sc
    c = pd.read_csv(TEST50_COHORT_CSV)
    rows = []
    for prefix, noise in (("F", "flat"), ("T", "slow")):
        for r in c.itertuples():
            name = f"{prefix}{r.source_id}_own4"
            res = sc.score_session(name, root)
            if res is None:
                print(f"  {name}: no output, skipped")
                continue
            rows.append({"noise": noise, "session": name, "source_id": r.source_id, "group": r.group,
                         "tau_group": r.tau_group, **res, **sc.tonic_metrics(name, root)})
    out = pd.DataFrame(rows)
    out.to_csv(FINAL_TEST_CSV, index=False)
    cols = ["f1_visible", "recall_visible", "precision", "f1_05", "recall_05", "precision_05", "timing_err_s", "amp_spearman",
            "tau_rise_err_pct", "tau_decay_err_pct", "phasic_taken", "windows_authors_rule"]
    f = lambda v: f"{v:.3f}"
    print(f"{FINAL_SETTING} on the 50 test participants (median), flat and slow noise")
    print("all:")
    print(out.groupby("noise")[cols].median().to_string(float_format=f))
    print("by tau group:")
    print(out.groupby(["tau_group", "noise"])[cols].median().to_string(float_format=f))
    print("by film group:")
    print(out.groupby(["group", "noise"])[cols].median().to_string(float_format=f))
    print("saved", FINAL_TEST_CSV.name)
    return out


def noise_spectrum(suffix="own4"):
    """Is the synthetic noise realistic?  Each tuning session's solver input against its source
    participant's real input through the same chain, band by band.

    full   the whole input (responses + tonic + noise): is the input as a whole realistic?
    quiet  stretches with no counted response: mostly noise, plus responses below 0.02 uS
    ratio = real / synthetic power.  About 1 everywhere: the synthetic noise is realistic.  Above 1
    in the slow bands: real data carry slow disturbances ours lack, so the synthetic test is
    optimistic for small pulses.
    On the first 18 tuning sessions; from round 6 the slow noise is estimated on the 50 selected
    instead (check50, SPECTRUM50_CURVES_CSV).
    """
    cutoff, fsy = VARIANTS[suffix]
    roles = split()
    tune = roles[roles.role == "tune"]
    src = pd.read_csv(ts.COHORT_CSV).set_index("session").source_id.astype(int)
    real_cache, rows, curves = {}, [], []
    for k, r in enumerate(tune.itertuples(), 1):
        pid = int(src[r.session])
        if pid not in real_cache:
            d = load(pid, every=1)
            real_cache[pid] = prepare(d["eda"], d["t"], RAW_FS, segment_of(d["marker"]), cutoff, fsy)["y"]
        syn = loadmat(tx.OUT / f"{r.session}_{suffix}.mat")["y"].ravel()
        sp_r, nq_r = spectra(real_cache[pid], fsy)
        sp_s, nq_s = spectra(syn, fsy)
        for kind in ("full", "quiet"):
            (f, pr), (_, ps) = sp_r[kind], sp_s[kind]
            curves += [{"session": r.session, "kind": kind, "f_hz": fi, "psd_real": a, "psd_synthetic": b}
                       for fi, a, b in zip(f, pr, ps)]
            for lo, hi in BANDS_HZ:
                a, b = band_power(f, pr, lo, hi), band_power(f, ps, lo, hi)
                rows.append({"session": r.session, "source_id": pid, "tau_group": r.tau_group,
                             "kind": kind, "band_hz": f"{lo}-{hi}", "power_real": a,
                             "power_synthetic": b, "ratio_real_syn": a / b if b > 0 else np.nan,
                             "quiet_windows_real": nq_r, "quiet_windows_syn": nq_s})
        if k % 6 == 0:
            print(f"  {k}/{len(tune)} sessions")
    out = pd.DataFrame(rows)
    out.to_csv(SPECTRUM_CSV, index=False)
    c = pd.DataFrame(curves)
    c.groupby(["kind", "f_hz"])[["psd_real", "psd_synthetic"]].median().reset_index() \
        .to_csv(SPECTRUM_CURVES_CSV, index=False)

    q = out.drop_duplicates("session")
    print(f"quiet 16 s windows per session, median: real {q.quiet_windows_real.median():.0f}, "
          f"synthetic {q.quiet_windows_syn.median():.0f}")
    for kind in ("full", "quiet"):
        g = out[out.kind == kind].groupby("band_hz", sort=False).ratio_real_syn
        t = pd.DataFrame({"pairs": g.count(), "median": g.median(), "q25": g.quantile(.25),
                          "q75": g.quantile(.75),
                          "share_real_higher": g.apply(lambda v: (v.dropna() > 1).mean())})
        print(f"\n{kind}: power ratio real / synthetic (pairs: sessions where both have quiet windows)")
        print(t.to_string(float_format=lambda v: f"{v:.2f}"))
    print("saved", SPECTRUM_CSV.name, SPECTRUM_CURVES_CSV.name)
    return out
