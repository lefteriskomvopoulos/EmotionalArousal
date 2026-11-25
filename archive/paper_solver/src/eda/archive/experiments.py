"""
Diagnostics that justified the hyperparameter choices in config.py.

These are not part of the analysis pipeline.  They are the experiments we ran to decide the
sparsity weight, the tonic window and the kernel bounds, kept in the repository so the choices
can be re-checked rather than taken on trust.

    lambda_sweep             one participant, several lambdas
    lambda_across_subjects   weak and strong responders, to pick ONE global lambda
    kernel_identifiability   fits the kernel for all usable participants, counts how many
                             end up pinned to a bound
    tonic_window_sweep       shows that the tonic window drives the fitted decay constant
"""
import numpy as np
import pandas as pd

from .. import config as cfg
from . import deconvolution as dec
from .. import figures
from ..dataset import film_mask, load
from ..quality import extreme_responders, usable_table


def _first_film_segment(d, phasic, win_s):
    """The first win_s seconds of the participant's first film block."""
    film = np.flatnonzero(film_mask(d["marker"]))
    i0 = film[0] if len(film) else 0
    return phasic[i0:i0 + int(win_s * cfg.DFS)]


def lambda_sweep(pid=None, win_s=120, lams=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)):
    """Impulse rate and fit against lambda, for one participant.

    Small lambda gives a perfect fit and implausibly many impulses; large lambda gives few
    impulses and a poor fit.  We want the value whose impulse rate is physiological.
    """
    from ..quality import choose_participants
    pid = pid or choose_participants()[0][0]

    d = dec.to_deconv_rate(load(pid))
    _, phasic = dec.split_tonic_phasic(d["eda"])
    seg = _first_film_segment(d, phasic, win_s)

    rates, r2s = [], []
    for lam in lams:
        tau_r, tau_d = dec.fit_kernel(seg, seg_s=win_s, lam=lam)
        h = dec.kernel(tau_r, tau_d)
        u = dec.deconvolve(seg, h, lam=lam)
        ev_t, ev_a = dec.extract_events(u)
        r2 = 1 - (seg - np.convolve(u, h)[:len(seg)]).var() / seg.var()
        rates.append(len(ev_t) / (win_s / 60))
        r2s.append(r2)
        print(f"  lambda {lam:5.2f} | tau {tau_r:4.2f}/{tau_d:5.2f} s | {len(ev_t):4d} impulses "
              f"({rates[-1]:5.1f}/min) | mean amp {ev_a.mean() if len(ev_a) else 0:.3f} µS "
              f"| R2 {r2:.3f}")

    figures.lambda_curve(pid, lams, rates, r2s, win_s)


def lambda_across_subjects(win_s=120, lams=(0.4, 0.8, 1.6, 3.2, 6.4, 12.8)):
    """The same sweep on the weakest and strongest responder of each group.

    One global lambda has to serve everyone, so it must put weak and strong responders alike in
    the physiological band.  This is the experiment that chose cfg.LAM.
    """
    curves = {}
    for pid, label in extreme_responders():
        d = dec.to_deconv_rate(load(pid))
        _, phasic = dec.split_tonic_phasic(d["eda"])
        seg = _first_film_segment(d, phasic, win_s)

        rates, r2s = [], []
        for lam in lams:
            tau_r, tau_d = dec.fit_kernel(seg, seg_s=win_s, lam=lam)
            h = dec.kernel(tau_r, tau_d)
            u = dec.deconvolve(seg, h, lam=lam)
            ev_t, _ = dec.extract_events(u)
            rates.append(len(ev_t) / (win_s / 60))
            r2s.append(1 - (seg - np.convolve(u, h)[:len(seg)]).var() / seg.var())

        curves[f"P{pid} {label}"] = (rates, r2s)
        print(f"P{pid:3d} {label:14s} rates " + " ".join(f"{r:5.1f}" for r in rates) +
              "  | R2 " + " ".join(f"{v:.2f}" for v in r2s))

    figures.lambda_across_subjects(lams, curves)


def kernel_identifiability(seg_s=300):
    """Fit the kernel for every usable participant and record whether tau hit a bound.

    A pinned value is not an estimate: it means the recording could not identify the kernel.
    Writes data/kernels_study6.csv.
    """
    rows = []
    for n, r in enumerate(usable_table().sort_values("burst_med_uS").itertuples(), 1):
        d = dec.to_deconv_rate(load(int(r.id)))
        _, phasic = dec.split_tonic_phasic(d["eda"])
        tau_r, tau_d = dec.fit_kernel(phasic, seg_s=seg_s)
        rise_pinned, decay_pinned = dec.tau_pinned(tau_r, tau_d)
        rows.append({"id": int(r.id), "group": r.group, "burst_med_uS": r.burst_med_uS,
                     "tau_rise": tau_r, "tau_decay": tau_d,
                     "rise_pinned": rise_pinned, "decay_pinned": decay_pinned})
        if n % 20 == 0:
            print(f"  {n} done")

    k = pd.DataFrame(rows)
    k.to_csv(cfg.KERNEL_CSV, index=False)

    print(f"\n{len(k)} participants | tau_rise pinned: {k.rise_pinned.sum()} "
          f"({k.rise_pinned.mean() * 100:.0f}%) | tau_decay pinned: {k.decay_pinned.sum()} "
          f"({k.decay_pinned.mean() * 100:.0f}%)")
    print("\nby responsiveness tercile (weakest first):")
    k["tercile"] = pd.qcut(k.burst_med_uS, 3, labels=["weak", "mid", "strong"])
    print(k.groupby("tercile", observed=True)[["rise_pinned", "decay_pinned"]]
          .mean().round(2).to_string())

    free = k[~k.rise_pinned & ~k.decay_pinned]
    print("\ntau distributions (unpinned only):")
    print(f"  n={len(free)}  tau_rise {free.tau_rise.min():.2f}-{free.tau_rise.max():.2f} "
          f"(median {free.tau_rise.median():.2f})  "
          f"tau_decay {free.tau_decay.min():.2f}-{free.tau_decay.max():.2f} "
          f"(median {free.tau_decay.median():.2f})")

    figures.kernel_identifiability(k)
    print("saved", cfg.KERNEL_CSV)


def tonic_window_sweep(pids=(5, 34, 99), windows=(10, 20, 30, 60, 120), seg_s=300):
    """Vary the tonic window and watch the fitted kernel.

    Our decay constant was once ~3x published values.  This sweep showed why: a wider window
    leaves more slow content in the phasic signal and the kernel stretches to absorb it.  It is
    the experiment that set cfg.TONIC_WIN_S.
    """
    print(f"{'pid':>4} {'win_s':>6} {'tau_r':>6} {'tau_d':>6} {'pinned':>7} {'imp/min':>8} {'R2':>6}")
    for pid in pids:
        d = dec.to_deconv_rate(load(pid))
        film = np.flatnonzero(film_mask(d["marker"]))
        for w in windows:
            _, phasic = dec.split_tonic_phasic(d["eda"], win_s=w)
            seg = phasic[film[0]:film[0] + int(seg_s * cfg.DFS)]
            tau_r, tau_d = dec.fit_kernel(seg, seg_s=seg_s)
            h = dec.kernel(tau_r, tau_d)
            u = dec.deconvolve(seg, h)
            ev_t, _ = dec.extract_events(u)
            r2 = 1 - (seg - np.convolve(u, h)[:len(seg)]).var() / seg.var()
            rise_p, decay_p = dec.tau_pinned(tau_r, tau_d)
            pinned = ("rise" if rise_p else "") + ("decay" if decay_p else "")
            print(f"{pid:>4} {w:>6} {tau_r:>6.2f} {tau_d:>6.2f} {pinned or '-':>7} "
                  f"{len(ev_t) / (seg_s / 60):>8.1f} {r2:>6.3f}")
