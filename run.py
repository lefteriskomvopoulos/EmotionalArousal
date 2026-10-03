"""
Command-line entry point.  Run from the project root:

    python -m eda.run <command> [participant_id]

Commands
    inspect            what is in the raw folder: files, channels, one block timeline
    quality            screen all 187 participants -> data/quality_study6.csv
    plot               raw EDA + affect slider for the chosen participants
    deconv  [pid]      deconvolution over a whole session + figure
    window  [pid]      one 120 s window through every stage + figure
    state   [pid]      deconvolution followed by state estimation + figure

Diagnostics that justified the hyperparameters (see experiments.py)
    sweep   [pid]      lambda sweep on one participant
    sweep-all          lambda sweep on weak and strong responders
    kernels            kernel identifiability across all usable participants
    tonic              tonic-window sweep
"""
import sys

import numpy as np

from . import config as cfg
from . import deconvolution as dec
from . import experiments
from . import figures
from . import state as st
from .dataset import baseline_mask, film_mask, load, read_header
from .quality import choose_participants, run_screen


def cmd_inspect():
    """Print the raw folder's contents, the channel header, and one block timeline."""
    files = sorted(cfg.RAW.glob("*.csv"))
    print(f"{len(files)} csv files, {len(list(cfg.RAW.glob('*_All.csv')))} participants")

    first = sorted(cfg.RAW.glob("*_All.csv"))[0]
    for k, v in read_header(first).items():
        if k.startswith(("Channel", "Data_Unit", "Data_Sample", "Data_Device")):
            print(f"  {k:17s} {v}")

    d = load(first.name.split("_")[1][1:])
    print(f"\nblock timeline of {first.name}:")
    for i in np.flatnonzero(d["marker"][1:] != d["marker"][:-1]) + 1:
        print(f"  {d['t'][i]:7.1f} s  ->  {d['marker'][i]}")


def cmd_plot():
    """Raw EDA and slider for a couple of usable participants per group."""
    records = [(pid, group, load(pid)) for pid, group in choose_participants()]
    for pid, group, d in records:
        print(f"P{pid} {group}: {d['t'][-1] / 60:.1f} min, "
              f"EDA {np.nanmin(d['eda']):.2f}–{np.nanmax(d['eda']):.2f} µS, "
              f"missing {np.isnan(d['eda']).mean() * 100:.1f}%")
    figures.raw_signals(records)


def cmd_deconv(pid=None):
    """Deconvolve a whole session and report per-block impulse statistics."""
    pid = pid or choose_participants()[0][0]
    d = dec.run(pid, load)
    t, mk, ev_t, ev_a = d["t"], d["marker"], d["ev_t"], d["ev_a"]

    minutes = (t[-1] - t[0]) / 60
    print(f"P{pid}: tau_rise {d['tau_rise']:.2f} s, tau_decay {d['tau_decay']:.2f} s | "
          f"{len(ev_t)} impulses ({len(ev_t) / minutes:.1f}/min) | fit R2 {d['r2']:.3f}")

    for name, mask in (("baseline", baseline_mask(mk)), ("films", film_mask(mk))):
        if mask.sum():
            sel = np.isin(((ev_t - t[0]) * cfg.DFS).astype(int), np.flatnonzero(mask))
            print(f"   {name:9s} {sel.sum():4d} impulses, "
                  f"{sel.sum() / (mask.sum() / cfg.DFS / 60):5.1f}/min, "
                  f"mean amplitude {ev_a[sel].mean() if sel.any() else 0:.3f}")

    figures.deconvolution_session(pid, d)


def cmd_window(pid=None, win_s=120):
    """Walk one short window through every stage, printing what each produces."""
    pid = pid or choose_participants()[0][0]
    d = dec.run(pid, load)

    film = np.flatnonzero(film_mask(d["marker"]))
    i0 = film[0] if len(film) else 0
    i1 = min(i0 + int(win_s * cfg.DFS), len(d["phasic"]))
    sl = slice(i0, i1)
    t = d["t"][sl]

    ev_t, ev_a = dec.extract_events(d["u"][sl], t0=t[0])
    d["win_ev_t"], d["win_ev_a"] = ev_t, ev_a

    r2 = 1 - (d["phasic"][sl] - d["fit"][sl]).var() / d["phasic"][sl].var()
    print(f"\nP{pid}, window {t[0]:.0f}-{t[-1]:.0f} s (block {d['marker'][i0]}), "
          f"{win_s} s at {cfg.DFS} Hz")
    print(f"  1 raw EDA        : {d['eda'][sl].min():.3f} to {d['eda'][sl].max():.3f} µS")
    print(f"  2 tonic          : {d['tonic'][sl].min():.3f} to {d['tonic'][sl].max():.3f} µS")
    print(f"  3 phasic         : {d['phasic'][sl].min():.3f} to {d['phasic'][sl].max():.3f} µS")
    print(f"  4 kernel         : tau_rise {d['tau_rise']:.2f} s, tau_decay {d['tau_decay']:.2f} s, "
          f"peak at {np.argmax(d['h']) / cfg.DFS:.1f} s")
    print(f"  5 impulses (u)   : {np.count_nonzero(d['u'][sl])} non-zero samples out of {i1 - i0}")
    print(f"  6 merged events  : {len(ev_t)} impulses, amplitudes "
          f"{ev_a.min():.3f}-{ev_a.max():.3f} µS, {len(ev_t) / (win_s / 60):.1f}/min")
    print(f"  7 check          : fit R2 = {r2:.3f}")

    figures.deconvolution_window(pid, d, sl, win_s)


def cmd_state(pid=99):
    """Deconvolution followed by state estimation, with the arousal figure."""
    d = dec.run(pid, load)
    s = st.run(d)
    t, mk, tb, xs = d["t"], d["marker"], s["t_bins"], s["state"]
    par = s["params"]

    print(f"P{pid}: {len(d['ev_t'])} impulses -> {len(xs)} bins of {cfg.BIN_S:.0f}s")
    print(f"  model: b0 {par['b0']:.2f}  g0 {par['g0']:.2f}  g1 {par['g1']:.2f}  "
          f"sr {par['sr']:.3f}  sv {par['sv']:.4f}")

    # which bins fall in the baseline and which in a film
    base = np.interp(tb, t, baseline_mask(mk).astype(float)) > .5
    film = np.interp(tb, t, film_mask(mk).astype(float)) > .5
    print(f"  mean arousal: baseline {xs[base].mean():+.3f}   films {xs[film].mean():+.3f}   "
          f"difference {xs[film].mean() - xs[base].mean():+.3f}")

    figures.arousal_state(pid, d, s)


COMMANDS = {
    "inspect": lambda pid: cmd_inspect(),
    "quality": lambda pid: run_screen(),
    "plot": lambda pid: cmd_plot(),
    "deconv": cmd_deconv,
    "window": cmd_window,
    "state": lambda pid: cmd_state(pid or 99),
    "sweep": experiments.lambda_sweep,
    "sweep-all": lambda pid: experiments.lambda_across_subjects(),
    "kernels": lambda pid: experiments.kernel_identifiability(),
    "tonic": lambda pid: experiments.tonic_window_sweep(),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    pid = int(argv[1]) if len(argv) > 1 else None
    COMMANDS[argv[0]](pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
