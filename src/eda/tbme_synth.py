"""
Synthetic sessions drawn from the real cohort, for testing the Amin & Faghih (2021) code.

N_SESSIONS sessions (S1, S2, ...), each built from the measured profile of one usable
participant picked at random (cohort_profile.py; no deconvolution involved).  Taking a whole
profile at once keeps traits that go together (large SCRs with a high tonic level, say)
together.  Each session is built at 100 Hz like our load and decimated to 2 Hz exactly as the
real export, so the published code receives the same kind of input it receives for a real
participant:

    x(t) = tonic(t) + sum_i a_i h(t - t_i) + noise(t),  rounded to the
           participant's quantisation step, with the participant's gaps put back

  h        Bateman kernel, peak 1 (so a_i is the response height in uS)
  tonic    the participant's floor curve, as measured
  t_i      at least 1 s apart; rate = participant's counted SCR rate x a correction factor,
           following their baseline / film / other mix
  a_i      log-normal: the participant's counted median and spread x a correction factor
  noise    white, the participant's measured SD x a correction factor
  no artefacts (see below)

The measures are not the true settings: peak detection merges overlapping SCRs (true rate >
counted), and the noise measure also picks up quantisation and response curvature.  So noise,
rate and size are corrected for the whole cohort, one factor per tau group (longer responses
overlap more), each the median ratio real / synthetic over the group's sessions:
  pass 1  all factors 1                -> noise factor
  pass 2  noise corrected              -> rate and size factors (counts depend on the noise,
                                          so they are set on signals that already carry it)
  final   all three corrected
A first version set all factors from one uncorrected pass and added artefacts; the final
sessions then counted 18% fewer, 24% larger peaks than their sources, because the factors had
been set on different signals.

Artefacts are left out.  The jump rule that counts them (cohort_profile.jumps) also fires on
SCR onsets, and far more often on ours (1.2/min from SCRs alone) than on the real recordings
(0.26/min in total): the Bateman response starts rising at full speed, real SCRs start more
gently.  The count is therefore not comparable between real and synthetic; it is still
reported.  The shape difference is invisible at the 2 Hz the published code works at.
Large real jumps (P99, 1126 s) are left for a separate stress test.

The time constants are the one thing raw data cannot give, so they are set by design.  The
drawn participants are sorted by SCR rate and dealt into three groups, so each group spans the
cohort:
  prior     drawn from the authors' priors, N(0.65, 0.21) / N(2.77, 0.52) s
  moderate  around 0.4 / 2.0 s or 1.0 / 4.0 s (alternating), +-10%
  far       around 1.2 / 5.0 s, +-10%

The check: the same model-free measures on the synthetic sessions next to the 119 real ones.

Saved: data/tbme_input/S<k>.mat (cluster input), data/tbme_synth/S<k>.npz (the truth),
data/tbme_synth_cohort.csv (one row per session: source participant, settings, measures),
data/tbme_synth_check.csv (real vs synthetic distributions).
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import cohort_profile as cp
from . import config as cfg
from . import tbme_export as tx
from .dataset import baseline_mask, film_mask

TRUTH_DIR = cfg.ROOT / "data" / "tbme_synth"
COHORT_CSV = cfg.ROOT / "data" / "tbme_synth_cohort.csv"
CHECK_CSV = cfg.ROOT / "data" / "tbme_synth_check.csv"
SEED = 2021
N_SESSIONS = 36                                            # 12 per tau group
KERNEL_S = 40
PRIOR = {"rise": (0.65, 0.21), "decay": (2.77, 0.52)}      # Amin & Faghih 2021
MODERATE = ((0.4, 2.0), (1.0, 4.0))
FAR = (1.2, 5.0)
CHECKED = ["scr_pm", "scr_amp_med_uS", "scr_amp_logsd", "noise_uS", "art_pm", "level_uS",
           "drift_uS"]


def kernel(tau_rise, tau_decay, fs=cfg.FS):
    t = np.arange(int(KERNEL_S * fs)) / fs
    h = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    return h / h.max()


def draw(prof, rng):
    """N_SESSIONS participants without replacement, each with a tau group and (tau_r, tau_d)."""
    picked = prof.sample(min(N_SESSIONS, len(prof)), random_state=rng.integers(2**31)).sort_values("scr_pm")
    out = []
    for k, row in enumerate(picked.itertuples()):
        g = ("prior", "moderate", "far")[k % 3]
        if g == "prior":
            tr = float(np.clip(rng.normal(*PRIOR["rise"]), 0.3, 1.2))
            td = float(np.clip(rng.normal(*PRIOR["decay"]), 1.8, 4.0))
        else:
            tr, td = MODERATE[(k // 3) % 2] if g == "moderate" else FAR
            tr, td = tr * rng.uniform(0.9, 1.1), td * rng.uniform(0.9, 1.1)
        out.append((row, g, tr, td))
    return out


def timeline(pid):
    """The real participant's 100 Hz time axis and block codes, rebuilt from their 2 Hz input."""
    real = loadmat(tx.OUT / f"P{pid}.mat")
    t2, mk2 = real["t"].ravel(), real["marker"].ravel()
    n = (len(t2) - 1) * tx.STEP + 1
    codes = np.where(np.isnan(mk2), "none", np.char.mod("%d", np.nan_to_num(mk2).astype(int)))
    return t2[0] + np.arange(n) / cfg.FS, np.repeat(codes, tx.STEP)[:n]


def rate_shape(mk, row):
    """Relative impulse rate per sample: the participant's baseline / film / other ratio."""
    m = np.ones(len(mk))
    if row.scr_pm > 0:
        m[baseline_mask(mk)] = row.scr_pm_base / row.scr_pm
        m[film_mask(mk)] = row.scr_pm_film / row.scr_pm
    return np.maximum(m, 0.05)


def build(row, t, mk, floor2, gaps, tau_rise, tau_decay, c_rate, c_amp, c_noise, seed, amp_cap=np.inf):
    """One synthetic 100 Hz segment -> dict (NaN at the participant's gaps), sizes <= amp_cap uS."""
    rng = np.random.default_rng(seed)
    fs, n = cfg.FS, len(t)
    dur = n / fs
    shape = rate_shape(mk, row)
    peak = shape.max()
    mean_gap = 60 / (row.scr_pm * c_rate * peak)
    times, t_ = [], rng.exponential(mean_gap)
    while t_ < dur:                                        # renewal, >= 1 s gaps, then thinned
        times.append(t_)
        t_ += cp.PEAK_GAP_S + rng.exponential(max(mean_gap - cp.PEAK_GAP_S, 0.05))
    times = np.array(times)
    times = times[rng.uniform(size=len(times)) < shape[(times * fs).astype(int)] / peak]
    sigma = row.scr_amp_logsd if row.scr_amp_logsd > 0 else 0.5   # nan when < 2 peaks counted
    amps = np.minimum(row.scr_amp_med_uS * c_amp * np.exp(sigma * rng.standard_normal(len(times))),
                      amp_cap)

    u = np.zeros(n)
    np.add.at(u, (times * fs).astype(int), amps)
    from scipy.signal import fftconvolve
    phasic = fftconvolve(u, kernel(tau_rise, tau_decay))[:n]

    tonic = np.interp(t, t[::tx.STEP], floor2)
    q = row.quant_uS if row.quant_uS > 0 else 0.01
    x = np.round((tonic + phasic + rng.standard_normal(n) * row.noise_uS * c_noise) / q) * q
    x[gaps] = np.nan
    return {"x": x, "tonic": tonic, "phasic": phasic, "ev_t": t[0] + times, "ev_a": amps}


def build_all(sessions, curves, c_rate, c_amp, c_noise):
    """Build and measure every session with the given per-group factors."""
    out = []
    for k, (row, g, tr, td) in enumerate(sessions, 1):
        pid = int(row.id)
        t, mk = timeline(pid)
        s = build(row, t, mk, curves[f"floor_P{pid}"], curves[f"gaps_P{pid}"], tr, td,
                  c_rate[g], c_amp[g], c_noise[g], SEED + k)
        m, _ = cp.measure(s["x"], t, mk)
        out.append((k, row, g, tr, td, t, mk, s, m))
    return out


def run():
    """Draw, correct (noise first, then rate and size), build, check and save the cohort."""
    prof = pd.read_csv(cp.PROFILE_CSV)
    curves = np.load(cp.CURVES_NPZ)
    sessions = draw(prof, np.random.default_rng(SEED))
    groups = ("prior", "moderate", "far")

    ones = {g: 1.0 for g in groups}
    ratio = lambda passed, g, col: float(np.median(
        [getattr(r, col) / m[col] for _, r, gg, *_, m in passed if gg == g and m[col] > 0]
        or [1.0]))                                         # [1.0]: one-participant smoke run
    # pass 1: everything uncorrected -> the noise factor
    p1 = build_all(sessions, curves, ones, ones, ones)
    c_noise = {g: ratio(p1, g, "noise_uS") for g in groups}
    # pass 2: noise corrected -> the rate and size factors, on signals with the final noise
    p2 = build_all(sessions, curves, ones, ones, c_noise)
    c_rate = {g: ratio(p2, g, "scr_pm") for g in groups}
    c_amp = {g: ratio(p2, g, "scr_amp_med_uS") for g in groups}
    print("correction factors (real / synthetic):",
          ", ".join(f"{g} noise x{c_noise[g]:.2f} rate x{c_rate[g]:.2f} size x{c_amp[g]:.2f}"
                    for g in groups))
    final = build_all(sessions, curves, c_rate, c_amp, c_noise)

    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    for f in list(tx.OUT.glob("S*.mat")) + list(TRUTH_DIR.glob("S*.npz")):
        f.unlink()                                         # no stale sessions from an older draw
    rows = []
    for k, row, g, tr, td, t, mk, s, m in final:
        t2 = t[::tx.STEP]
        y2, gaps2 = tx.fill(s["x"][::tx.STEP], t2)
        tx.save(f"S{k}", y2, t2, mk[::tx.STEP], gaps2.sum())
        np.savez_compressed(TRUTH_DIR / f"S{k}.npz", ev_t=s["ev_t"], ev_a=s["ev_a"],
                            tonic2=s["tonic"][::tx.STEP], phasic2=s["phasic"][::tx.STEP],
                            tau_rise=tr, tau_decay=td, t2=t2, source_id=int(row.id))
        rows.append({"session": f"S{k}", "source_id": int(row.id), "group": row.group,
                     "tau_group": g, "tau_rise": tr, "tau_decay": td,
                     "c_rate": c_rate[g], "c_amp": c_amp[g], "c_noise": c_noise[g],
                     "n_true": len(s["ev_t"]),
                     "rate_true_pm": len(s["ev_t"]) / row.dur_min,
                     "amp_med_true_uS": float(np.median(s["ev_a"])) if len(s["ev_a"]) else np.nan,
                     **{f"real_{c}": getattr(row, c) for c in CHECKED},
                     **{f"syn_{c}": m[c] for c in CHECKED}})
    c = pd.DataFrame(rows)
    c.to_csv(COHORT_CSV, index=False)

    qs = [.1, .25, .5, .75, .9]
    check = pd.DataFrame([{"measure": col, "set": name, **{f"q{int(q * 100)}": v.quantile(q)
                                                         for q in qs}}
                          for col in CHECKED
                          for name, v in (("real_119", prof[col]),
                                          ("synthetic", c[f"syn_{col}"]))])
    check.to_csv(CHECK_CSV, index=False)
    print("real (all usable) vs synthetic, quantiles 10/25/50/75/90%:")
    print(check.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("paired, each session vs its source participant: synthetic / real, median [10-90%]"
          " (artefacts: difference per min, not matched; level: difference in uS)")
    for col in CHECKED:
        r, y = c[f"real_{col}"], c[f"syn_{col}"]
        v = y - r if col in ("art_pm", "level_uS") else y / r
        print(f"  {col:15s} {v.median():.3g} [{v.quantile(.1):.3g}, {v.quantile(.9):.3g}]")
    print("per tau group (median):")
    print(c.groupby("tau_group")[["tau_rise", "tau_decay", "rate_true_pm", "syn_scr_pm",
                                  "real_scr_pm"]].median()
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print("saved", COHORT_CSV.name, CHECK_CSV.name, f"{TRUTH_DIR.name}/S*.npz",
          f"{tx.OUT.name}/S*.mat")
    return c


# ------------------------------------------------------------------ preprocessing test: level
SHIFT_TARGETS = {"m0": 0.0, "m2": 2.0}     # suffix -> minimum of the shifted signal, uS
LOW_LEVEL_uS = 0.4                         # the 8 sessions below this were the 8 least stable


def shift_inputs():
    """S<k>_m0.mat / S<k>_m2.mat for every session copied from a low-level participant.

    Same signal, shifted by a constant so its minimum is 0 uS (offset removed) or 2 uS (a
    physiological level); the truth file is shared, the cluster runs them as new sessions.
    """
    c = pd.read_csv(COHORT_CSV)
    low = c[c.real_level_uS < LOW_LEVEL_uS].sort_values("real_level_uS")
    names = []
    for r in low.itertuples():
        S = loadmat(tx.OUT / f"{r.session}.mat")
        y, t = S["y"].ravel(), S["t"].ravel()
        mk = np.where(np.isnan(S["marker"].ravel()), "none",
                      np.char.mod("%d", np.nan_to_num(S["marker"].ravel()).astype(int)))
        for suf, target in SHIFT_TARGETS.items():
            ys, shift = tx.set_floor(y, target)
            tx.save(f"{r.session}_{suf}", ys, t, mk, int(S["filled"].item()))
            names.append(f"{r.session}_{suf}")
        print(f"{r.session} (source P{r.source_id}, level {r.real_level_uS:.2f} uS, "
              f"min {y.min():.2f} uS): shifted copies {', '.join(SHIFT_TARGETS)}")
    print(f"saved {len(names)} inputs in {tx.OUT.name}/: {' '.join(names)}")
    return names


# ------------------------------------------------------------------ preprocessing test: rate
RATE_SESSIONS = {"far": ("S9", "S15", "S24", "S33"),     # same four as the window test
                 "prior": None}                          # filled by rate_inputs: 4, level >= 0.4
RATE_FSY = 4                                             # Hz, twice the published 2 Hz


def rebuilt(wanted=None, amp_caps=None):
    """Yield the 100 Hz sessions: (name, profile row, tau group, tau_r, tau_d, t, mk, build dict).

    The 100 Hz signals are not stored, so they are rebuilt: same draw, same seeds, same
    correction factors (read from the saved cohort table).  Every rebuilt session is checked
    against its saved 2 Hz file, so the truth files stay valid.  amp_caps {pid: uS} then rebuilds
    the session with its sizes capped, everything else unchanged (same random numbers).
    """
    prof = pd.read_csv(cp.PROFILE_CSV)
    curves = np.load(cp.CURVES_NPZ)
    cohort = pd.read_csv(COHORT_CSV).set_index("session")
    for k, (row, g, tr, td) in enumerate(draw(prof, np.random.default_rng(SEED)), 1):
        sname = f"S{k}"
        if wanted is not None and sname not in wanted:
            continue
        c = cohort.loc[sname]
        pid = int(row.id)
        t, mk = timeline(pid)
        s = build(row, t, mk, curves[f"floor_P{pid}"], curves[f"gaps_P{pid}"], tr, td,
                  c.c_rate, c.c_amp, c.c_noise, SEED + k)
        y2, _ = tx.fill(s["x"][::tx.STEP], t[::tx.STEP])
        if not np.allclose(y2, loadmat(tx.OUT / f"{sname}.mat")["y"].ravel()):
            raise RuntimeError(f"{sname}: rebuilt signal differs from the saved one")
        if amp_caps is not None:
            s = build(row, t, mk, curves[f"floor_P{pid}"], curves[f"gaps_P{pid}"], tr, td,
                      c.c_rate, c.c_amp, c.c_noise, SEED + k, amp_cap=amp_caps[pid])
        yield sname, row, g, tr, td, t, mk, s


def rate_inputs(fsy=RATE_FSY, n_prior=4):
    """S<k>_f<fsy>.mat: chosen sessions rebuilt and sampled at fsy instead of 2 Hz.

    Preprocessing step 1 applies as for real data: a negative minimum is shifted to 0.
    """
    cohort = pd.read_csv(COHORT_CSV).set_index("session")
    prior = cohort[(cohort.tau_group == "prior") & (cohort.real_level_uS >= LOW_LEVEL_uS)]
    wanted = set(RATE_SESSIONS["far"]) | set(prior.sort_values("real_scr_pm").index[
        np.linspace(0, len(prior) - 1, n_prior).round().astype(int)])
    step = int(cfg.FS // fsy)
    names = []
    for sname, row, g, tr, td, t, mk, s in rebuilt(wanted):
        pid = int(row.id)
        yf, gaps = tx.fill(s["x"][::step], t[::step])
        if yf.min() < 0:
            yf, _ = tx.set_floor(yf, 0.0)
        tx.save(f"{sname}_f{fsy}", yf, t[::step], mk[::step], gaps.sum(), fs=fsy)
        names.append(f"{sname}_f{fsy}")
        print(f"{sname} ({g}, tau {tr:.2f} / {td:.2f} s, source P{pid}): {len(yf)} samples "
              f"at {fsy} Hz, matches the saved 2 Hz session")
    print(f"saved {len(names)} inputs in {tx.OUT.name}/: {' '.join(names)}")
    return names


# ------------------------------------------------------------------ the paper's preprocessing
PAPER_SESSIONS = ("S4", "S16", "S25", "S34", "S9", "S15", "S24", "S33")   # 4 prior, 4 far


def paper_inputs(sessions=PAPER_SESSIONS):
    """S<k>_pp.mat + truth: sessions through tbme_export.paper_preprocess, real and synthetic alike.

    Response sizes are capped at the largest SCR counted on the source participant, because the
    uncapped log-normal tail reached 15 uS against at most ~4 uS in real inputs.  Tonic and
    phasic truth go through the same filter, so they stay comparable with the input; event
    times do not move, the filter having no delay.
    """
    peaks = pd.read_csv(cp.PEAKS_CSV)
    caps = peaks.groupby("id").amp_uS.max().to_dict()
    for sname, row, g, tr, td, t, mk, s in rebuilt(set(sessions), caps):
        y100, n_jumps = tx.paper_preprocess(s["x"], t)
        y2, t2, mk2 = y100[::tx.STEP], t[::tx.STEP], mk[::tx.STEP]
        shift = 0.0
        if y2.min() < 0:
            y2, shift = tx.set_floor(y2, 0.0)
        tx.save(f"{sname}_pp", y2, t2, mk2, np.isnan(s["x"][::tx.STEP]).sum())
        np.savez_compressed(TRUTH_DIR / f"{sname}_pp.npz", ev_t=s["ev_t"], ev_a=s["ev_a"],
                            tonic2=tx.low_pass(s["tonic"])[::tx.STEP] + shift,
                            phasic2=tx.low_pass(s["phasic"])[::tx.STEP],
                            tau_rise=tr, tau_decay=td, t2=t2, source_id=int(row.id))
        print(f"{sname}_pp ({g}, source P{int(row.id)}): {n_jumps} jumps, largest size "
              f"{s['ev_a'].max():.2f} uS (cap {caps[int(row.id)]:.2f}), shifted +{shift:.3f} uS")
