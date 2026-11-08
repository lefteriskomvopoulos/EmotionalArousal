"""
Automatic screening of the 187 recordings.

Four criteria of our own, plus the dataset authors' own EDA flag.  All are applied identically to
every participant and were fixed before any analysis was run.  The result is one row per
participant in data/quality_study6.csv, with a `usable` column.

187 -> 119 usable: neutral 29, negative 29, positive 32, mixed 29.
"""
import numpy as np
import pandas as pd

from . import config as cfg
from .dataset import baseline_mask, film_mask, groups_from_metadata, load


def quality_metrics(d):
    """Descriptive numbers for one recording, computed on the cfg.FS Hz signal.

    These are deliberately cheap and independent of the deconvolution: the screen must not
    depend on the analysis it is screening for.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    eda, aff, mk = d["eda"], d["affect"], d["marker"]
    ok = ~np.isnan(eda)
    x = np.interp(np.arange(len(eda)), np.flatnonzero(ok), eda[ok])   # fill gaps, for filtering only

    b, a = butter(2, 0.05 / (cfg.FS / 2), "high")      # drop anything slower than ~20 s
    phasic = filtfilt(b, a, x)
    peaks, _ = find_peaks(phasic, prominence=0.02, distance=cfg.FS)   # >= 0.02 uS, >= 1 s apart

    step = np.diff(x)                                   # sample-to-sample change
    minutes = d["t"][-1] / 60
    in_base, in_film = baseline_mask(mk), film_mask(mk)
    rate = lambda m: np.isin(peaks, np.flatnonzero(m)).sum() / max(m.sum() / cfg.FS / 60, 1e-9)

    return {
        "dur_min": minutes,
        "eda_min": np.nanmin(eda), "eda_max": np.nanmax(eda),
        "neg_frac": np.mean(eda < 0),       # negative values are an amplifier offset, harmless
        "missing_frac": np.mean(~ok),       # empty samples in the file
        "bursts_per_min": len(peaks) / minutes,
        "bursts_base_pm": rate(in_base), "bursts_film_pm": rate(in_film),
        "burst_med_uS": float(np.median(phasic[peaks])) if len(peaks) else 0.0,
        "hf_jitter_nS": np.std(step) * 1000,            # noise floor
        "spike_frac": np.mean(np.abs(step) > cfg.SPIKE_JUMP_uS),   # electrode noise
        "slider_moves": int(np.sum(np.diff(aff[~np.isnan(aff)]) != 0)),
    }


def is_usable(q):
    """Vectorised rejection rule over the quality table.

    A negative offset alone is NOT a reason to reject: it is a constant amplifier shift and
    disappears when the tonic component is subtracted.
    """
    return ~((q.spike_frac > cfg.MAX_SPIKE_FRAC)
             | (q.missing_frac > cfg.MAX_MISSING_FRAC)
             | (q.bursts_per_min < cfg.MIN_BURSTS_PER_MIN)
             | (q.burst_med_uS < cfg.MIN_MEDIAN_BURST_uS)
             | q.meta_eda_flag)


def run_screen(verbose=True):
    """Screen every participant and write cfg.QUALITY_CSV. Returns the table."""
    rows = []
    for i, (pid, (group, flagged)) in enumerate(sorted(groups_from_metadata().items()), 1):
        rows.append({"id": pid, "group": group, "meta_eda_flag": flagged,
                     **quality_metrics(load(pid))})
        if verbose and i % 20 == 0:
            print(f"  {i} done")

    q = pd.DataFrame(rows)
    q["usable"] = is_usable(q)
    cfg.QUALITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(cfg.QUALITY_CSV, index=False)
    if verbose:
        print(q.groupby("group").usable.agg(usable="sum", total="count"))
        print("saved", cfg.QUALITY_CSV)
    return q
