"""
The new 50: the 50 drawn participants with 6 replaced by 6 of the other 47, a change made in stage
2.  Recomputes the real-data numbers the report and the slides quote, for the original 50 (so it
can be seen that they reproduce the reported values) and for the new 50, side by side.  Reads only
the frozen run's output for both sets (hpc/own/final, hpc/own/extra) and the profile tables.

    PYTHONPATH=src python -m eda.new50 --out 12,34,56,78,90,111 --in 1,11,33,35,38,39

-> data/new50_numbers.csv     one row per number: original 50, new 50
   data/new50_participants.csv one row per participant of either set
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import cohort_profile as cp
from . import config as cfg
from . import selection
from .runs import (EXTRA_FITS_CSV, EXTRA_OUT, EXTRA_PULSES_CSV, FINAL6_SETTING, FINAL_OUT, FITS_CSV,
                   PULSES_CSV)

NUMBERS_CSV = cfg.ROOT / "data" / "new50_numbers.csv"
PEOPLE_CSV = cfg.ROOT / "data" / "new50_participants.csv"


def person(pid, root, pulses, wins, peaks):
    """One participant's real-data numbers, from the frozen run's output."""
    sig = pd.read_csv(root / f"P{pid}_own4_signals.csv", usecols=["time_s", "y", "y_rec", "marker"])
    y, rec = sig.y.to_numpy(), sig.y_rec.to_numpy()
    w = wins[wins.id == pid]
    p = pulses[pulses.id == pid]
    minutes = (w.t_end.max() - w.t_start.min()) / 60
    base_min = (sig.marker == -1).sum() * np.median(np.diff(sig.time_s)) / 60
    big, base = p.height_uS >= 0.05, p.block == "baseline"
    pk = peaks[(peaks.id == pid) & (pd.to_numeric(peaks.marker, errors="coerce") == -1)]
    return {"id": pid, "r2": 1 - np.sum((y - rec) ** 2) / np.sum((y - y.mean()) ** 2),
            "pulses_pm": len(p) / minutes, "pulses05_pm": big.sum() / minutes,
            "base_pulses_pm": base.sum() / base_min, "base_pulses05_pm": (base & big).sum() / base_min,
            "base_counted05_pm": (pk.amp_uS >= 0.05).sum() / base_min,
            "windows": len(w), "windows_l2_at_cap": int((w.lambda2 >= 0.95 * w.lambda2_cap).sum()),
            "n_pulses": len(p)}


def numbers(t):
    """The quoted numbers for one set of 50 (t: one row per participant)."""
    rho_all = spearmanr(t.pulses_pm, t.counted_pm)[0]
    rho_05 = spearmanr(t.pulses05_pm, t.counted_pm)[0]
    return {"participants": len(t),
            "neutral / positive / negative / mixed": "/".join(str(int((t.group == g).sum()))
                                                            for g in ("neutral", "positive", "negative", "mixed")),
            "fit R2, median": round(t.r2.median(), 4),
            "pulses per min, median": round(t.pulses_pm.median(), 1),
            "pulses >= 0.05 uS per min, median": round(t.pulses05_pm.median(), 1),
            "counted peaks per min (>= 0.02 uS), median": round(t.counted_pm.median(), 1),
            "pulses / counted, ratio of medians": round(t.pulses_pm.median() / t.counted_pm.median(), 2),
            "rest: pulses per min, median": round(t.base_pulses_pm.median(), 1),
            "rest: pulses >= 0.05 uS per min, median": round(t.base_pulses05_pm.median(), 1),
            "rest: counted >= 0.05 uS per min, median": round(t.base_counted05_pm.median(), 1),
            "rest: counted >= 0.05 uS per min, range": f"{t.base_counted05_pm.min():.1f}-{t.base_counted05_pm.max():.1f}",
            "rank corr. pulse rate vs counted rate (all)": round(rho_all, 2),
            "rank corr. pulse rate vs counted rate (>= 0.05 uS)": round(rho_05, 2),
            "windows with GCV lambda2 at its cap": f"{t.windows_l2_at_cap.sum() / t.windows.sum():.0%}",
            "pulses in all": int(t.n_pulses.sum())}


def main(out_ids, in_ids):
    sel = pd.read_csv(selection.SELECTION_CSV)
    drawn = set(sel.loc[sel.selected.astype(bool), "id"].astype(int))
    pool = set(sel.loc[sel.stage2_pass.astype(bool) & ~sel.selected.astype(bool), "id"].astype(int))
    bad_out, bad_in = set(out_ids) - drawn, set(in_ids) - pool
    if bad_out or bad_in or len(out_ids) != len(in_ids):
        raise SystemExit(f"--out must be drawn participants, --in from the other 47, as many of each; "
                         f"not drawn: {sorted(bad_out)}, not in the 47: {sorted(bad_in)}")
    new = (drawn - set(out_ids)) | set(in_ids)

    group = sel.set_index("id").group
    counted = pd.read_csv(cp.PROFILE_CSV).set_index("id").scr_pm
    peaks = pd.read_csv(cp.PEAKS_CSV)
    final = (FINAL_OUT / FINAL6_SETTING, pd.read_csv(PULSES_CSV), pd.read_csv(FITS_CSV))
    extra = (EXTRA_OUT / FINAL6_SETTING, pd.read_csv(EXTRA_PULSES_CSV), pd.read_csv(EXTRA_FITS_CSV))
    rows = []
    for pid in sorted(drawn | new):
        r = person(pid, *(final if pid in drawn else extra), peaks)
        rows.append({**r, "group": group[pid], "counted_pm": counted[pid],
                     "in_original": pid in drawn, "in_new": pid in new})
    t = pd.DataFrame(rows)
    t.to_csv(PEOPLE_CSV, index=False)
    out = pd.DataFrame({"original 50": numbers(t[t.in_original]), "new 50": numbers(t[t.in_new])})
    out.index.name = "number"
    out.to_csv(NUMBERS_CSV)
    print(f"out: {sorted(out_ids)}  in: {sorted(in_ids)}")
    print(out.to_string())
    print("saved", NUMBERS_CSV.name, PEOPLE_CSV.name)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--out", required=True, help="the 6 drawn participants taken out, comma-separated ids")
    a.add_argument("--in", dest="in_", required=True, help="the 6 put in, from the other 47")
    args = a.parse_args()
    main([int(x) for x in args.out.split(",")], [int(x) for x in args.in_.split(",")])
