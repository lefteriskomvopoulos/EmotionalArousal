"""
Every figure the project produces.  All of them are written to results/figures/.

Kept separate from the analysis so that the analysis modules import nothing from matplotlib and
can be used from a notebook or a test without a display.
"""
import matplotlib
matplotlib.use("Agg")                    # no display needed; must come before pyplot
import matplotlib.pyplot as plt
import numpy as np

from . import config as cfg
from .dataset import block_changes


def _save(fig, name):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.FIG_DIR / name
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)
    return out


def _mark_blocks(ax, t, marker):
    """Vertical lines at block boundaries, labelled with the block code."""
    for i in block_changes(marker):
        ax.axvline(t[i], color="k", lw=.5, alpha=.4)


def raw_signals(records, name=None):
    """One panel per participant: raw EDA, the affect slider, and the block boundaries.

    `records` is a list of (participant_id, group_label, data_dict).
    """
    fig, axes = plt.subplots(len(records), 1, figsize=(14, 3.2 * len(records)), squeeze=False)
    for ax, (pid, group, d) in zip(axes[:, 0], records):
        ax.plot(d["t"], d["eda"], lw=.7, color=cfg.COLOR_SIGNAL)
        ax.set_ylabel(f"P{pid} {group}\nEDA (µS)")

        ax2 = ax.twinx()
        ax2.plot(d["t"], d["affect"], lw=.9, color=cfg.COLOR_TONIC, alpha=.8)
        ax2.set_ylim(0, 11)
        ax2.set_ylabel("affect 1–10", color=cfg.COLOR_TONIC)

        for i in block_changes(d["marker"]):
            ax.axvline(d["t"][i], color="k", lw=.6, alpha=.5)
            if d["marker"][i] != "none":
                ax.text(d["t"][i] + 3, ax.get_ylim()[1], d["marker"][i], fontsize=8, va="top")

    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("POPANE Study 6 — raw EDA, affect slider, block codes "
                 "(−1 baseline; 1xx neutral, 2xx negative, 3xx positive)")
    fig.tight_layout()
    return _save(fig, name or f"study6_raw_{len(records)}participants.png")


def deconvolution_session(pid, dec):
    """Whole session: raw EDA with the tonic estimate, phasic with the reconstruction, impulses."""
    t, mk = dec["t"], dec["marker"]
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    ax[0].plot(t, dec["eda"], lw=.6, color=cfg.COLOR_SIGNAL, label=f"EDA ({cfg.DFS} Hz)")
    ax[0].plot(t, dec["tonic"], lw=1.4, color=cfg.COLOR_TONIC, label="tonic estimate")
    ax[0].set_ylabel("µS"); ax[0].legend(fontsize=8, loc="upper left")

    ax[1].plot(t, dec["phasic"], lw=.6, color=cfg.COLOR_SIGNAL, label="phasic")
    ax[1].plot(t, dec["fit"], lw=1.0, color=cfg.COLOR_EVENTS, ls="--",
               label=f"re-convolved fit (R²={dec['r2']:.2f})")
    ax[1].set_ylabel("µS"); ax[1].legend(fontsize=8, loc="upper left")

    ax[2].vlines(dec["ev_t"], 0, dec["ev_a"], color=cfg.COLOR_EVENTS, lw=1.2)
    ax[2].set_ylabel("impulse amplitude"); ax[2].set_xlabel("time (s)")

    for a in ax:
        _mark_blocks(a, t, mk)
    fig.suptitle(f"P{pid} — deconvolution: τ_rise {dec['tau_rise']:.2f} s, "
                 f"τ_decay {dec['tau_decay']:.2f} s, {len(dec['ev_t'])} impulses")
    fig.tight_layout()
    return _save(fig, f"study6_deconv_P{pid}.png")


def deconvolution_window(pid, dec, sl, win_s):
    """One short window through the four stages, small enough to read off the page."""
    t = dec["t"][sl]
    ev_t, ev_a = dec["win_ev_t"], dec["win_ev_a"]

    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(t, dec["eda"][sl], color=cfg.COLOR_SIGNAL, lw=1)
    ax[0].plot(t, dec["tonic"][sl], color=cfg.COLOR_TONIC, lw=1.5)
    ax[0].set_ylabel("1 raw + tonic\nµS")

    ax[1].plot(t, dec["phasic"][sl], color=cfg.COLOR_SIGNAL, lw=1)
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_ylabel("2 phasic\nµS")

    ax[2].vlines(ev_t, 0, ev_a, color=cfg.COLOR_EVENTS, lw=1.5)
    ax[2].set_ylabel("3 impulses u\namplitude")

    ax[3].plot(t, dec["phasic"][sl], color=cfg.COLOR_SIGNAL, lw=1, label="phasic")
    ax[3].plot(t, dec["fit"][sl], color=cfg.COLOR_EVENTS, ls="--", lw=1.2, label="h * u")
    ax[3].legend(fontsize=8); ax[3].set_ylabel("4 check\nµS"); ax[3].set_xlabel("time (s)")

    fig.suptitle(f"P{pid} — one {win_s}s window through the pipeline "
                 f"(τ_rise {dec['tau_rise']:.2f}s, τ_decay {dec['tau_decay']:.2f}s, "
                 f"{len(ev_t)} impulses)")
    fig.tight_layout()
    return _save(fig, f"study6_demo_P{pid}.png")


def arousal_state(pid, dec, st):
    """Phasic signal, recovered impulses, and the estimated arousal with its uncertainty band."""
    t, mk = dec["t"], dec["marker"]
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    ax[0].plot(t, dec["phasic"], lw=.6, color=cfg.COLOR_SIGNAL)
    ax[0].set_ylabel("phasic EDA\nµS")

    ax[1].vlines(dec["ev_t"], 0, dec["ev_a"], color=cfg.COLOR_EVENTS, lw=1)
    ax[1].set_ylabel("impulses\nµS")

    tb, xs, sd = st["t_bins"], st["state"], st["state_sd"]
    ax[2].fill_between(tb, xs - sd, xs + sd, color=cfg.COLOR_EVENTS, alpha=.25)
    ax[2].plot(tb, xs, color=cfg.COLOR_EVENTS, lw=1.5)
    ax[2].axhline(0, color="k", lw=.5)
    ax[2].set_ylabel("estimated arousal\n(± 1 SD)"); ax[2].set_xlabel("time (s)")

    ax2 = ax[2].twinx()
    ax2.plot(t, dec["affect"], color=cfg.COLOR_TONIC, lw=.9, alpha=.7)
    ax2.set_ylim(0, 11); ax2.set_ylabel("slider", color=cfg.COLOR_TONIC)

    for a in ax:
        _mark_blocks(a, t, mk)
    fig.suptitle(f"P{pid} — deconvolution to arousal state "
                 f"(τ {dec['tau_rise']:.2f}/{dec['tau_decay']:.2f}s, "
                 f"{len(dec['ev_t'])} impulses, {cfg.BIN_S:.0f}s bins)")
    fig.tight_layout()
    return _save(fig, f"study6_state_P{pid}.png")


def lambda_curve(pid, lams, rates, r2s, win_s):
    """Impulse rate and fit quality against the sparsity weight, for one participant."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(lams, rates, "o-", color=cfg.COLOR_EVENTS)
    ax[0].axhspan(5, 15, color=cfg.COLOR_EVENTS, alpha=.12)
    ax[0].set_xlabel("λ"); ax[0].set_ylabel("impulses per minute")
    ax[0].set_title("shaded = physiological range")
    ax[1].semilogx(lams, r2s, "o-", color=cfg.COLOR_SIGNAL)
    ax[1].set_xlabel("λ"); ax[1].set_ylabel("fit R²"); ax[1].set_ylim(0, 1.02)
    fig.suptitle(f"P{pid} — λ sweep on a {win_s}s film window")
    fig.tight_layout()
    return _save(fig, f"study6_lambda_sweep_P{pid}.png")


def lambda_across_subjects(lams, curves):
    """The same sweep for several participants, to choose one global λ."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (rates, r2s) in curves.items():
        style = "-" if "high" in name else "--"
        ax[0].semilogx(lams, rates, style, marker="o", ms=3, label=name)
        ax[1].semilogx(lams, r2s, style, marker="o", ms=3)
    ax[0].axhspan(5, 15, color=cfg.COLOR_EVENTS, alpha=.12)
    ax[0].set_xlabel("λ"); ax[0].set_ylabel("impulses per minute"); ax[0].set_ylim(0, 50)
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xlabel("λ"); ax[1].set_ylabel("fit R²"); ax[1].set_ylim(0, 1.02)
    fig.suptitle("Choosing one global λ: impulse rate and fit across weak and strong responders")
    fig.tight_layout()
    return _save(fig, "study6_lambda_across_subjects.png")


def kernel_identifiability(k):
    """Fitted time constants against responsiveness, with the bounds drawn in.

    Points sitting exactly on a dashed line were not identified: the optimiser ran into the bound.
    """
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, col, rng, name in ((ax[0], "tau_rise", cfg.TAU_RISE_RANGE, "τ_rise"),
                              (ax[1], "tau_decay", cfg.TAU_DECAY_RANGE, "τ_decay")):
        a.scatter(k.burst_med_uS, k[col], s=14, c=cfg.COLOR_SIGNAL, alpha=.7)
        for b in rng:
            a.axhline(b, color=cfg.COLOR_TONIC, ls="--", lw=1)
        a.set_xlabel("median burst size (µS)  →  responsiveness")
        a.set_ylabel(f"{name} (s)")
        a.set_title(f"{name}: dashed = bounds")
    fig.suptitle("Is the kernel identifiable? Fitted τ vs responsiveness, all usable participants")
    fig.tight_layout()
    return _save(fig, "study6_kernel_boundaries.png")
