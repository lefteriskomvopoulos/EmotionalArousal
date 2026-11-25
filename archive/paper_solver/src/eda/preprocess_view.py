"""
What the solver is given, real and synthetic, side by side.

Real: the raw 1000 Hz recording against the 2 Hz input written by tbme_export.  Synthetic: the
100 Hz session against its 2 Hz input.  Both are then described by cohort_profile.measure, the
same yardstick used to build the synthetic cohort, and by their power spectra at 100 Hz.

Saved: results/figures/preprocess_real.png, preprocess_synth.png, preprocess_compare.png
       data/preprocess_compare.csv
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend, find_peaks, welch

from . import cohort_profile as cp
from . import config as cfg
from . import figures
from . import tbme_export as tx
from . import tbme_synth as ts
from .dataset import load
from .tau_measure import REAL_IDS

COMPARE_CSV = cfg.ROOT / "data" / "preprocess_compare.csv"
PAPER_DIR = cfg.ROOT / "external" / "Tonic-Phasic-Decomposition-TBME"
PAPER_MATS = ("ss_8_14_18.mat", "ss_8_14_18_auditory_stimulation.mat")   # the first is what their scripts load
PAPER_CHECK_CSV = cfg.ROOT / "data" / "paper_input_check.csv"
PAPER_SPECTRUM_CSV = cfg.ROOT / "data" / "paper_input_spectrum.csv"
RAW_EVERY = int(1000 // cfg.FS)
WELCH_N = 2 ** 13


def worst_jump(x):
    """The largest move over 0.1 s, uS."""
    k = int(cp.JUMP_S * cfg.FS)
    return float(np.abs(x[k:] - x[:-k]).max())


def describe(name, kind, x, t, mk, t_raw, x_raw, shift, tau=None):
    """One real or synthetic session -> everything the figures and the table need."""
    m = loadmat(tx.OUT / f"{name}.mat")
    t_in, y_in = m["t"].ravel(), m["y"].ravel()
    row, _ = cp.measure(x, t, mk)
    xf, _ = tx.fill(x, t)
    jump = worst_jump(xf)
    f, p = welch(detrend(xf), fs=cfg.FS, nperseg=WELCH_N)
    return {"name": name, "kind": kind, "tau": tau, "t_raw": t_raw, "x_raw": x_raw,
            "seg": (t[0], t[-1]), "t_in": t_in, "y_in": y_in, "f": f, "p": p,
            "row": {"name": name, "kind": kind, "shift_uS": shift, "input_min_uS": y_in.min(),
                    "input_max_uS": y_in.max(), "worst_jump_uS": jump, **row}}


def paper_input_check():
    """What the authors' own 100 Hz input looks like, against ours before and after the filter.

    Their scripts downsample ss_8_14_18.mat, which their readme calls preprocessed.  Its spectrum
    shows which filter it went through, and so whether
    tbme_export.low_pass reproduces it.  Fewer distinct values than samples means an untouched
    quantised recording; almost every value distinct means it has been filtered.
    """
    from scipy.io import loadmat
    path = next(PAPER_DIR / m for m in PAPER_MATS if (PAPER_DIR / m).exists())
    print("reading", path.name)
    ss = loadmat(path, variable_names=["ss"], squeeze_me=True, struct_as_record=False)["ss"]
    first = np.atleast_1d(ss)[0]
    print("ss fields:", first._fieldnames, "| sc fields:", np.atleast_1d(first.sc)[0]._fieldnames)
    psd, rows = [], []
    for k, s in enumerate(np.atleast_1d(ss), 1):
        y = np.asarray(np.atleast_1d(s.sc)[0].y, float).ravel()
        if y.size < WELCH_N:
            continue
        f, p = welch(detrend(y), fs=cfg.FS, nperseg=WELCH_N)
        psd.append(p)
        rows.append({"subject": k, "minutes": y.size / cfg.FS / 60, "min_uS": y.min(), "max_uS": y.max(),
                     "distinct_share": len(np.unique(y)) / y.size,
                     "jumps": len(find_peaks(np.abs(np.diff(y)), prominence=tx.JUMP_PROMINENCE)[0])})
    raw, filt = [], []
    for pid in REAL_IDS:
        d = load(pid)
        sl = tx.segment(d)
        x, _ = tx.fill(d["eda"][sl], d["t"][sl])
        rows.append({"subject": f"P{pid}", "minutes": x.size / cfg.FS / 60, "min_uS": x.min(),
                     "max_uS": x.max(), "distinct_share": len(np.unique(x)) / x.size,
                     "jumps": len(find_peaks(np.abs(np.diff(x)), prominence=tx.JUMP_PROMINENCE)[0])})
        raw.append(welch(detrend(x), fs=cfg.FS, nperseg=WELCH_N)[1])
        filt.append(welch(detrend(tx.low_pass(x)), fs=cfg.FS, nperseg=WELCH_N)[1])
    spec = pd.DataFrame({"f_hz": f, "theirs": np.median(psd, 0), "ours_raw": np.median(raw, 0),
                         "ours_filtered": np.median(filt, 0)})
    spec.to_csv(PAPER_SPECTRUM_CSV, index=False)
    each = pd.DataFrame(rows)
    each.to_csv(PAPER_CHECK_CSV, index=False)
    figures.paper_spectrum(spec, "paper_input_spectrum.png")
    print(each.assign(theirs=each.subject.map(lambda v: not str(v).startswith("P")))
          .groupby("theirs")[["minutes", "min_uS", "max_uS", "distinct_share", "jumps"]].median()
          .to_string(float_format=lambda v: f"{v:.4g}"))
    print("median power (uS^2/Hz) near each frequency:")
    near = [(spec.f_hz - v).abs().idxmin() for v in (0.1, 0.5, 1, 2, 3, 5, 10, 20, 40)]
    print(spec.loc[near].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("saved", PAPER_CHECK_CSV.name, PAPER_SPECTRUM_CSV.name)


def run():
    shifts = pd.read_csv(tx.SHIFT_CSV).set_index("id").shift_uS
    real = []
    for pid in REAL_IDS:
        d = load(pid, every=1)
        d100 = {k: v[::RAW_EVERY] for k, v in d.items()}
        sl = tx.segment(d100)
        real.append(describe(f"P{pid}", "real", d100["eda"][sl], d100["t"][sl], d100["marker"][sl],
                             d["t"], d["eda"], float(shifts[pid])))

    cohort = pd.read_csv(ts.COHORT_CSV).sort_values("real_scr_pm")
    wanted = set(cohort.session.iloc[np.linspace(0, len(cohort) - 1, 5).round().astype(int)])
    syn = [describe(name, "synthetic", s["x"], t, mk, t, s["x"], 0.0, (group, tr, td))
           for name, row, group, tr, td, t, mk, s in ts.rebuilt(wanted)]

    tab = pd.DataFrame([r["row"] for r in real + syn])
    tab.to_csv(COMPARE_CSV, index=False)
    figures.preprocess_steps(real, "preprocess_real.png", "raw 1000 Hz")
    figures.preprocess_steps(syn, "preprocess_synth.png", "100 Hz")
    figures.preprocess_compare(real, syn, "preprocess_compare.png")
    print(tab[["name", "input_min_uS", "input_max_uS", "shift_uS", "worst_jump_uS", "noise_uS",
               "art_pm", "art_size_med_uS", "scr_pm", "scr_amp_med_uS", "drift_uS"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("saved", COMPARE_CSV.name)
    return tab
