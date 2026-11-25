"""
Real participants through the Amin & Faghih (2021) published code: summary and figures.

Reads the cluster output (P<id>_signals/_events/_windows.csv) from hpc/real/output and, with no
ground truth available, reports what can be checked on real data:

  fit             R2 and RMSE of the reconstruction; worst window R2
  tau             median and spread over windows (expected: at the priors, see the synthetic tests)
  windows         share chosen by the authors' rule
  events          per minute, raw and after the size cut-off (config.TBME_CUT_U), overall,
                  in the baseline and in the films
  plausibility    next to the model-free SCR count of the same participant (cohort_profile):
                  the synthetic tests put the true impulse rate at ~1.1-1.5x the counted one
  preprocessing   the shift added before the run (data/tbme_input_shifts.csv)

Saved: data/tbme_real_summary.csv and results/figures/tbme_real_P<id>.png.
"""
import numpy as np
import pandas as pd

from . import cohort_profile as cp
from . import config as cfg
from . import figures
from . import tbme_export as tx

OUTPUT = cfg.ROOT / "hpc" / "real" / "output"
SUMMARY_CSV = cfg.ROOT / "data" / "tbme_real_summary.csv"
ZOOM_S = 120


def summarise(pid, out_dir=OUTPUT, cut=cfg.TBME_CUT_U, suffix=""):
    name = f"P{pid}{suffix}"
    sig = pd.read_csv(out_dir / f"{name}_signals.csv")
    ev = pd.read_csv(out_dir / f"{name}_events.csv")
    win = pd.read_csv(out_dir / f"{name}_windows.csv")
    prof = pd.read_csv(cp.PROFILE_CSV).set_index("id").loc[pid]
    shifts = pd.read_csv(tx.SHIFT_CSV).set_index("id").shift_uS if tx.SHIFT_CSV.exists() else {}

    y, yr = sig.y.to_numpy(), sig.y_rec.to_numpy()
    mk = sig.marker.to_numpy()
    minutes = len(y) / tx.FSY / 60
    base_min = np.sum(mk == -1) / tx.FSY / 60
    film_min = np.sum(mk >= 100) / tx.FSY / 60
    ev_mk = np.interp(ev.time_s, sig.time_s, np.nan_to_num(mk, nan=0))
    keep = ev.amplitude.to_numpy() >= cut
    in_base = np.isclose(ev_mk, -1)
    in_film = ev_mk >= 100

    # zoom on the film stretch with the most kept events, so the figure shows the busy part
    starts = np.arange(sig.time_s.iloc[0], sig.time_s.iloc[-1] - ZOOM_S, 10)
    t_keep = ev.time_s.to_numpy()[keep]
    t0 = starts[np.argmax([np.sum((t_keep >= s) & (t_keep < s + ZOOM_S)) for s in starts])]
    figures.tbme_reconstruction(name, sig, ev, cut, (t0, t0 + ZOOM_S))

    return {
        "id": pid, "group": prof.group, "shift_uS": float(shifts.get(pid, np.nan)),
        "r2": 1 - np.sum((y - yr) ** 2) / np.sum((y - y.mean()) ** 2),
        "rmse_uS": np.sqrt(np.mean((y - yr) ** 2)), "r2_worst_window": win.r2.min(),
        "tau_rise_median": win.tau_rise.median(), "tau_decay_median": win.tau_decay.median(),
        "tau_decay_spread": win.tau_decay.max() - win.tau_decay.min(),
        "windows_authors_rule": (win.selection == "authors").mean(),
        "events_pm_raw": len(ev) / minutes, "events_pm_kept": keep.sum() / minutes,
        "kept_pm_baseline": (keep & in_base).sum() / max(base_min, 1e-9),
        "kept_pm_films": (keep & in_film).sum() / max(film_min, 1e-9),
        "counted_scr_pm": prof.scr_pm, "counted_scr_pm_base": prof.scr_pm_base,
        "counted_scr_pm_film": prof.scr_pm_film,
        "kept_over_counted": (keep.sum() / minutes) / prof.scr_pm,
        "solver_s": win.seconds.sum(),
    }


def run(pids=None, out_dir=OUTPUT):
    pids = pids or sorted(int(p.name[1:].split("_")[0]) for p in out_dir.glob("P*_events.csv"))
    rows = []
    for pid in pids:
        if not (out_dir / f"P{pid}_events.csv").exists():
            print(f"no output yet for P{pid}")
            continue
        rows.append(summarise(pid, out_dir))
    if not rows:
        return None
    sm = pd.DataFrame(rows)
    sm.to_csv(SUMMARY_CSV, index=False)
    print(sm.set_index("id").T.to_string(float_format=lambda v: f"{v:.3g}"))
    print("saved", SUMMARY_CSV.name, "and results/figures/tbme_real_P*.png")
    return sm
