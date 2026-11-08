"""
What the usable recordings look like, measured without any deconvolution.

The synthetic sessions our deconvolution is tuned on must look like OUR cohort, not like one or
two participants, and they must not be built from the output of the method they test.  So
every participant is described by model-free numbers taken on exactly the segment the method
sees (baseline start to last film, see tbme_export.segment):

  duration, missing samples   as recorded
  quantisation step           smallest non-zero change between samples
  tonic level and drift       a rolling 5th-percentile floor, 30 s window, smoothed over 30 s;
                              level = its median, drift = its 5-95% range
  SCR rate and size           trough-to-peak at 2 Hz: peaks with prominence >= 0.02 uS and at
                              least 1 s apart, size = rise from the lowest point up to 5 s before
                              (the conventional SCR amplitude; 0.02 uS = twice the resolution)
  noise                       SD at 100 Hz of the signal minus its 1 s moving average, away
                              from jumps
  artefacts                   episodes where the signal moves > 0.05 uS in 0.1 s (the quality
                              screen's spike rule), merged within 1 s; rate, size, duration

measure() is applied unchanged to the synthetic sessions (tbme_synth.py), so real and synthetic
are compared on the same yardstick, including its known blind spots (peak detection misses
SCRs that overlap).

Everything is saved:
  data/profile_study6.csv         one row per participant
  data/profile_study6_peaks.csv   every detected SCR: pid, time, size, block
  data/profile_study6_curves.npz  per participant: the 2 Hz floor and the 100 Hz gap positions
and the plain 2 Hz file is written in the same pass (data/tbme_input/P<pid>.mat): the synthetic
sessions copy its time axis and block codes (tbme_synth.timeline).
"""
import numpy as np
import pandas as pd

from . import config as cfg
from . import tbme_export as tx
from .dataset import baseline_mask, film_mask, load

PROFILE_CSV = cfg.ROOT / "data" / "profile_study6.csv"
PEAKS_CSV = cfg.ROOT / "data" / "profile_study6_peaks.csv"
CURVES_NPZ = cfg.ROOT / "data" / "profile_study6_curves.npz"

PEAK_MIN_uS = cfg.MIN_EVENT_uS   # 0.02 uS
PEAK_GAP_S = 1.0                 # the authors' minimum_peak_distance
RISE_MAX_S = 5.0                 # an SCR rises over 1-4 s
FLOOR_PCT, FLOOR_WIN_S = 5, 30
NOISE_WIN_S = 1.0
JUMP_S, JUMP_uS = 0.1, cfg.SPIKE_JUMP_uS
EPISODE_GAP_S = 1.0


def scr_peaks(y, fs=tx.FSY):
    """Trough-to-peak SCRs on a gap-free signal -> (peak indices, sizes in uS)."""
    from scipy.signal import find_peaks
    p, pr = find_peaks(y, prominence=PEAK_MIN_uS, distance=max(int(PEAK_GAP_S * fs), 1),
                       wlen=int(2 * RISE_MAX_S * fs) + 1)
    return p, y[p] - y[pr["left_bases"]]


def floor(y, fs=tx.FSY):
    """Rolling low percentile, smoothed: the level the bursts sit on."""
    from scipy.ndimage import percentile_filter, uniform_filter1d
    w = int(FLOOR_WIN_S * fs)
    return uniform_filter1d(percentile_filter(y, FLOOR_PCT, size=w, mode="nearest"), w,
                            mode="nearest")


def jumps(x, fs=cfg.FS):
    """Artefact episodes on a gap-free 100 Hz signal -> (sizes uS, durations s, flagged mask)."""
    k = int(JUMP_S * fs)
    dd = np.abs(x[k:] - x[:-k])
    idx = np.flatnonzero(dd > JUMP_uS)
    flag = np.zeros(len(x), bool)
    if not len(idx):
        return np.array([]), np.array([]), flag
    cuts = np.flatnonzero(np.diff(idx) > EPISODE_GAP_S * fs) + 1
    sizes, durs = [], []
    for ep in np.split(idx, cuts):
        sizes.append(dd[ep].max())
        durs.append((ep[-1] - ep[0] + k) / fs)
        flag[max(ep[0] - fs, 0):ep[-1] + k + fs] = True        # +-1 s, kept out of the noise
    return np.array(sizes), np.array(durs), flag


def measure(x, t, mk):
    """All model-free numbers for one segment at cfg.FS Hz (NaN = missing) -> (row, extras).

    The 2 Hz signal is taken exactly as tbme_export does it (every STEP-th sample, gaps
    interpolated), so the SCR and floor numbers describe what the published code receives.
    """
    fs = cfg.FS
    y2, g2 = tx.fill(x[::tx.STEP], t[::tx.STEP])
    mk2 = mk[::tx.STEP]
    xf, gaps = tx.fill(x, t)

    pk, amp = scr_peaks(y2)
    fl = floor(y2)
    size, dur, near = jumps(xf)
    from scipy.ndimage import uniform_filter1d
    r = xf - uniform_filter1d(xf, int(NOISE_WIN_S * fs), mode="nearest")
    ok = ~near & ~gaps
    steps = np.abs(np.diff(x[~np.isnan(x)]))
    minutes = len(y2) / tx.FSY / 60
    in_b, in_f = baseline_mask(mk2), film_mask(mk2)
    per_min = lambda m: np.isin(pk, np.flatnonzero(m)).sum() / max(m.sum() / tx.FSY / 60, 1e-9)

    row = {
        "dur_min": minutes,
        "missing_frac": gaps.mean(),
        "quant_uS": float(np.round(steps[steps > 1e-9].min(), 6)) if (steps > 1e-9).any() else np.nan,
        "level_uS": float(np.median(fl)),
        "drift_uS": float(np.percentile(fl, 95) - np.percentile(fl, 5)),
        "scr_pm": len(pk) / minutes,
        "scr_pm_base": per_min(in_b),
        "scr_pm_film": per_min(in_f),
        "scr_amp_med_uS": float(np.median(amp)) if len(amp) else np.nan,
        "scr_amp_logsd": float(np.std(np.log(amp))) if len(amp) > 1 else np.nan,
        "scr_amp_p90_uS": float(np.percentile(amp, 90)) if len(amp) else np.nan,
        "noise_uS": float(r[ok].std()) if ok.any() else np.nan,
        "art_pm": len(size) / minutes,
        "art_size_med_uS": float(np.median(size)) if len(size) else np.nan,
        "art_dur_med_s": float(np.median(dur)) if len(dur) else np.nan,
    }
    extras = {"peaks_t": t[::tx.STEP][pk], "peaks_amp": amp, "peaks_mk": mk2[pk],
              "floor2": fl, "gaps100": np.flatnonzero(gaps)}
    return row, extras


def run(pids=None):
    """Measure every usable participant (or `pids`), export their 2 Hz inputs, save all."""
    q = pd.read_csv(cfg.QUALITY_CSV)
    q = q[q.usable]
    if pids is not None:
        q = q[q.id.isin(pids)]
    rows, peaks, curves = [], [], {}
    for i, (pid, group) in enumerate(zip(q.id.astype(int), q.group), 1):
        d = load(pid)
        tx.export(pid, d)
        sl = tx.segment(d)
        row, ex = measure(d["eda"][sl], d["t"][sl], d["marker"][sl])
        rows.append({"id": pid, "group": group, **row})
        peaks.append(pd.DataFrame({"id": pid, "time_s": ex["peaks_t"], "amp_uS": ex["peaks_amp"],
                                   "marker": ex["peaks_mk"]}))
        curves[f"floor_P{pid}"] = ex["floor2"]
        curves[f"gaps_P{pid}"] = ex["gaps100"]
        if i % 10 == 0 or i == len(q):                         # save as we go
            pd.DataFrame(rows).to_csv(PROFILE_CSV, index=False)
            pd.concat(peaks).to_csv(PEAKS_CSV, index=False)
            np.savez_compressed(CURVES_NPZ, **curves)
            print(f"  {i}/{len(q)} measured, saved")

    p = pd.DataFrame(rows)
    show = ["dur_min", "missing_frac", "quant_uS", "level_uS", "drift_uS", "scr_pm",
            "scr_pm_base", "scr_pm_film", "scr_amp_med_uS", "scr_amp_logsd", "noise_uS",
            "art_pm", "art_size_med_uS"]
    print(p[show].describe(percentiles=[.1, .5, .9]).T[["min", "10%", "50%", "90%", "max"]]
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print("by group (median):")
    print(p.groupby("group")[["scr_pm", "scr_amp_med_uS", "level_uS", "noise_uS", "art_pm"]]
          .median().to_string(float_format=lambda v: f"{v:.3g}"))
    print("saved", PROFILE_CSV.name, PEAKS_CSV.name, CURVES_NPZ.name)
    return p
