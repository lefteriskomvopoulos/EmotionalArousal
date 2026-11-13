"""
Small input files, one .mat per participant or session, in the format src/own/own_run.m reads.

The cluster only needs these files (~50 kB each), never the 19 GB of raw csv.  The format is the
one of the Amin & Faghih (IEEE TBME 68:1726, 2021) scripts, hence the name: fs, y, t, marker.

export() writes each usable participant's plain 2 Hz segment, data/tbme_input/P<pid>.mat.  The
deconvolution does not run on it (preprocess.prepare builds its input from the 1000 Hz
file); it is the time axis and block codes that the synthetic sessions copy (tbme_synth.timeline),
and cohort_profile writes it in the same pass.

Ours, stated for the report:
  - only one continuous segment is kept, from the first sample of the five-minute resting
    baseline to the last sample of the last film.  The hypotheses use nothing else, and the
    recordings run 21-29 min against ~17 min of baseline + films.  t keeps the original
    timestamps, so events still line up with the full recording.
  - missing samples (under 5% by the quality screen) are filled by linear interpolation,
    because the solver has no notion of a gap.
  - a recording that goes below 0 uS (a POPANE amplifier offset) is shifted up by a constant
    so its minimum is 0: on synthetic sessions negative signals made tau jump between its
    bounds window to window, and the shift removed that (shift test, black-box branch).  Logged
    in data/tbme_input_shifts.csv.
"""
import numpy as np
import pandas as pd

from . import config as cfg
from .dataset import baseline_mask, film_mask, load

FSY = 2                                           # Hz, the published signal rate
STEP = int(cfg.FS // FSY)                         # plain decimation
OUT = cfg.ROOT / "data" / "tbme_input"


def segment(d):
    """Indices into the cfg.FS arrays of d whose every STEP-th sample is the exported 2 Hz segment.

    The 2 Hz signal is cut from the first baseline sample to the last film sample, after
    decimation, so the files already run on the cluster (P99, P5) stay exactly reproducible.
    """
    mk = d["marker"][::STEP]
    i0, i1 = np.flatnonzero(baseline_mask(mk))[0], np.flatnonzero(film_mask(mk))[-1] + 1
    return slice(i0 * STEP, (i1 - 1) * STEP + 1)


def fill(y, t):
    """Missing samples filled by linear interpolation -> (filled copy, gap mask)."""
    gaps = np.isnan(y)
    if gaps.any():
        y = y.copy()
        y[gaps] = np.interp(t[gaps], t[~gaps], y[~gaps])
    return y, gaps


def set_floor(y, target_uS):
    """Shift y by a constant so its minimum is target_uS -> (shifted y, shift added).

    The model has no offset term: its spline tonic C q carries the whole level, under C q <= y
    and a ridge penalty lambda1 ||q||^2, so the level is not neutral.  On synthetic sessions
    copied from participants whose level sits near or below zero (POPANE amplifier offsets),
    tau jumped between its bounds from window to window, and a constant shift removed that.
    Phasic part and impulses are untouched by a constant.
    """
    c = target_uS - np.min(y)
    return y + c, c


SHIFT_CSV = cfg.ROOT / "data" / "tbme_input_shifts.csv"


def log_shift(pid, shift):
    """One row per exported participant: the constant added (0 unless the signal went negative)."""
    row = pd.DataFrame({"id": [pid], "shift_uS": [shift]})
    log = pd.concat([pd.read_csv(SHIFT_CSV).query("id != @pid"), row]) if SHIFT_CSV.exists() else row
    log.sort_values("id").to_csv(SHIFT_CSV, index=False)


def save(name, y, t, mk, n_filled, fs=FSY):
    """data/tbme_input/<name>.mat in the format src/own/own_run.m reads (rate fs, Hz)."""
    from scipy.io import savemat
    marker = pd.to_numeric(pd.Series(mk), errors="coerce").to_numpy(float)   # "none" -> NaN
    OUT.mkdir(parents=True, exist_ok=True)
    savemat(OUT / f"{name}.mat", {"fs": fs, "y": y[:, None], "t": t[:, None],
                                  "marker": marker[:, None], "filled": int(n_filled)})


def export(pid, d=None):
    """Write data/tbme_input/P<pid>.mat with y (uS), t (s), marker (block code) at 2 Hz."""
    d = load(pid) if d is None else d             # cfg.FS = 100 Hz
    sl = segment(d)
    y, t, mk = d["eda"][sl][::STEP], d["t"][sl][::STEP], d["marker"][sl][::STEP]
    inside = sorted(set(mk) - {"-1"} - {m for m in set(mk) if m[:1] in "123"})
    y, gaps = fill(y, t)
    shift = 0.0
    if y.min() < 0:
        y, shift = set_floor(y, 0.0)
    save(f"P{pid}", y, t, mk, gaps.sum())
    log_shift(pid, shift)
    print(f"P{pid}: {t[0]:.0f}-{t[-1]:.0f} s, {len(y)} samples ({len(y) / FSY / 60:.1f} min), "
          f"{gaps.sum()} filled, shifted +{shift:.3f} uS, other blocks inside: "
          f"{inside or 'none'} -> {OUT.name}/P{pid}.mat")
