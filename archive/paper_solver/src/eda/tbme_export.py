"""
Input files for the Amin & Faghih (IEEE TBME 68:1726, 2021) code, one small .mat per participant.

The authors' scripts downsample a 100 Hz signal to 2 Hz with MATLAB's plain downsample, but that
signal was already preprocessed (supplement, section C): large discontinuities removed and a
low-pass filter applied.  export() leaves those two steps out and paper_export() includes them
(paper_preprocess).  Either way the cluster only needs these files (~50 kB each), never the
19 GB of raw csv.

Ours, stated for the report:
  - only one continuous segment is kept, from the first sample of the five-minute resting
    baseline to the last sample of the last film.  The hypotheses use nothing else, and the
    recordings run 21-29 min against ~17 min of baseline + films.  t keeps the original
    timestamps, so events still line up with the full recording.
  - missing samples (under 5% by the quality screen) are filled by linear interpolation,
    because the published solver has no notion of a gap.
  - a recording that goes below 0 uS (a POPANE amplifier offset) is shifted up by a constant
    so its minimum is 0: on synthetic sessions negative signals made tau jump between its
    bounds window to window; the shift removed that (tbme_synth.shift_test).  Logged in
    data/tbme_input_shifts.csv.
"""
import numpy as np
import pandas as pd

from . import config as cfg
from .dataset import baseline_mask, film_mask, load

FSY = 2                                           # Hz, the authors' signal rate
STEP = int(cfg.FS // FSY)                         # plain decimation, as their downsample()
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

    Preprocessing candidate.  The published model has no offset term: its spline tonic C q
    carries the whole level, under C q <= y and a ridge penalty lambda1 ||q||^2, so the level
    is not neutral.  On synthetic sessions copied from participants whose level sits near or
    below zero (POPANE amplifier offsets), tau jumped between its bounds from window to window;
    tbme_synth.shift_test checks whether a constant shift removes that.  Phasic part and
    impulses are untouched by a constant.
    """
    c = target_uS - np.min(y)
    return y + c, c


# Amin & Faghih (2021) supplement, section C: 0.1 prominence, 0.5 s patch, cubic B-spline, low-pass
# with a 0.5 Hz cutoff, down-sampling to 2 Hz.  The paper gives no filter type or order.  We use
# the FIR of their LowPassFilter (Hamming window, gain 1 at DC, signal extended by its end values,
# delay removed) with order 640: at 100 Hz their order 64 lasts 0.65 s and cuts near 1 Hz, not the
# stated 0.5 Hz.  Their published scripts and solver do not filter at all (docs/EXPERIMENTS.md, part 5).
JUMP_PROMINENCE = 0.1
JUMP_PATCH_S = 0.5
FIR_ORDER = 640
FIR_CUTOFF_HZ = 0.5


def remove_jumps(x, t):
    """Peaks of prominence 0.1 on |diff(x)|: 0.5 s around each replaced by a cubic B-spline."""
    from scipy.interpolate import make_interp_spline
    from scipy.signal import find_peaks
    pk = find_peaks(np.abs(np.diff(x)), prominence=JUMP_PROMINENCE)[0]
    half = int(JUMP_PATCH_S * cfg.FS / 2)
    drop = np.zeros(len(x), bool)
    for p in pk:
        drop[max(p + 1 - half, 0):p + 1 + half] = True
    if drop.any():
        x = x.copy()
        x[drop] = make_interp_spline(t[~drop], x[~drop], k=3)(t[drop])
    return x, len(pk)


def low_pass(x):
    """Zero-delay FIR low-pass at cfg.FS with a true 0.5 Hz cutoff (see the constants above)."""
    from scipy.signal import firwin, lfilter
    b = firwin(FIR_ORDER + 1, FIR_CUTOFF_HZ / (cfg.FS / 2), window="hamming")
    d = FIR_ORDER // 2
    out = lfilter(b, 1.0, np.r_[np.full(d, x[0]), x, np.full(d, x[-1])])
    return out[2 * d:]


def paper_preprocess(x, t):
    """100 Hz segment -> (preprocessed 100 Hz signal, jumps removed), real and synthetic alike."""
    x, _ = fill(x, t)                             # ours: the published method assumes no gaps
    x, n_jumps = remove_jumps(x, t)
    return low_pass(x), n_jumps


SHIFT_CSV = cfg.ROOT / "data" / "tbme_input_shifts.csv"


def log_shift(pid, shift):
    """One row per exported participant: the constant added (0 unless the signal went negative)."""
    row = pd.DataFrame({"id": [pid], "shift_uS": [shift]})
    log = pd.concat([pd.read_csv(SHIFT_CSV).query("id != @pid"), row]) if SHIFT_CSV.exists() else row
    log.sort_values("id").to_csv(SHIFT_CSV, index=False)


def save(name, y, t, mk, n_filled, fs=FSY):
    """data/tbme_input/<name>.mat in the format src/tbme/tbme_run.m reads (rate fs, Hz)."""
    from scipy.io import savemat
    marker = pd.to_numeric(pd.Series(mk), errors="coerce").to_numpy(float)   # "none" -> NaN
    OUT.mkdir(parents=True, exist_ok=True)
    savemat(OUT / f"{name}.mat", {"fs": fs, "y": y[:, None], "t": t[:, None],
                                  "marker": marker[:, None], "filled": int(n_filled)})


def export(pid, d=None):
    """Write data/tbme_input/P<pid>.mat with y (uS), t (s), marker (block code) at 2 Hz."""
    d = load(pid) if d is None else d             # cfg.FS = 100 Hz, like the authors' data
    sl = segment(d)
    y, t, mk = d["eda"][sl][::STEP], d["t"][sl][::STEP], d["marker"][sl][::STEP]
    inside = sorted(set(mk) - {"-1"} - {m for m in set(mk) if m[:1] in "123"})
    y, gaps = fill(y, t)
    shift = 0.0
    if y.min() < 0:                               # preprocessing step 1 (tbme_synth.shift_test)
        y, shift = set_floor(y, 0.0)
    save(f"P{pid}", y, t, mk, gaps.sum())
    log_shift(pid, shift)
    print(f"P{pid}: {t[0]:.0f}-{t[-1]:.0f} s, {len(y)} samples ({len(y) / FSY / 60:.1f} min), "
          f"{gaps.sum()} filled, shifted +{shift:.3f} uS, other blocks inside: "
          f"{inside or 'none'} -> {OUT.name}/P{pid}.mat")


def paper_export(pid):
    """data/tbme_input/P<pid>_pp.mat: the paper's preprocessing, then the 2 Hz samples of export().

    The negative-level shift stays: it is ours, not the paper's, and needed by the solver.
    """
    d = load(pid)
    sl = segment(d)
    y100, n_jumps = paper_preprocess(d["eda"][sl], d["t"][sl])
    y, t, mk = y100[::STEP], d["t"][sl][::STEP], d["marker"][sl][::STEP]
    shift = 0.0
    if y.min() < 0:
        y, shift = set_floor(y, 0.0)
    save(f"P{pid}_pp", y, t, mk, np.isnan(d["eda"][sl][::STEP]).sum())
    print(f"P{pid}_pp: {n_jumps} jumps removed, shifted +{shift:.3f} uS, {len(y)} samples")


def export_all(pid=None):
    """One participant, or every participant the quality screen kept."""
    if pid is not None:
        return export(pid)
    q = pd.read_csv(cfg.QUALITY_CSV)
    for p in q.loc[q["usable"], "id"]:
        export(int(p))
