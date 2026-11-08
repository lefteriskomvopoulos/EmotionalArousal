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


def _save(fig, name):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.FIG_DIR / name
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)
    return out


def preprocess_stages(item, name):
    """One session, whole timeline, one row per preprocessing step.

    Left: the signal after the step.  Right: what the step changed, on its own scale, or the
    fact that it changed nothing.  Bottom: 60 s around the steepest rise, before and after the
    low-pass, with the 2 Hz samples the solver receives.
    """
    rows, z = item["rows"], item["zoom"]
    fig = plt.figure(figsize=(16, 1.9 * len(rows) + 3.2))
    grid = fig.add_gridspec(len(rows) + 1, 2, width_ratios=[2, 1], height_ratios=[1] * len(rows) + [1.6])
    first = None
    for k, r in enumerate(rows):
        left = fig.add_subplot(grid[k, 0], sharex=first)
        right = fig.add_subplot(grid[k, 1], sharex=left)
        first = first or left
        left.plot(r["t"], r["x"], lw=.5, color=cfg.COLOR_SIGNAL)
        left.set_ylabel("µS", fontsize=8)
        left.set_title(r["label"], loc="left", fontsize=10)
        if "seg" in r:
            left.axvspan(*r["seg"], color="0.85", alpha=.5, lw=0)
        if r["change"] is not None and np.nanmax(np.abs(r["change"])) > 0:
            right.plot(r.get("t_change", r["t"]), r["change"], lw=.5, color=cfg.COLOR_TONIC)
            right.set_ylabel("µS", fontsize=8)
        else:
            right.set_yticks([])
        right.set_title(r["note"], loc="left", fontsize=9)
        for a in (left, right):
            a.tick_params(labelsize=8)
            if k < len(rows) - 1:
                a.tick_params(labelbottom=False)
    bottom = fig.add_subplot(grid[-1, :])
    bottom.plot(z["t"], z["before"], lw=.7, color=cfg.COLOR_SIGNAL, label="before the low-pass (100 Hz)")
    bottom.plot(z["t"], z["low"], lw=1.4, color=cfg.COLOR_TONIC, label="after the low-pass")
    bottom.plot(z["t2"], z["y2"], "o", ms=4, color=cfg.COLOR_EVENTS, label="2 Hz samples the solver gets")
    bottom.set_title("zoom: 60 s around the steepest rise", loc="left", fontsize=10)
    bottom.set_xlabel("time (s)")
    bottom.set_ylabel("µS", fontsize=8)
    bottom.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    fig.suptitle(f"{item['name']}: every preprocessing step", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def filter_response(resp, cutoff, nyquist, name):
    """Gain of the low-pass against frequency, with its cutoff and the 2 Hz input's limit."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(resp.f_hz, resp.gain, lw=2, color=cfg.COLOR_SIGNAL)
    for f, label in ((cutoff, f"cutoff {cutoff} Hz"), (nyquist, f"2 Hz input's limit, {nyquist:g} Hz")):
        ax.axvline(f, color="0.6", lw=1, ls=":")
        ax.text(f + .03, .92, label, fontsize=9)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("gain (1 = kept, 0 = removed)")
    ax.set_title("Low-pass filter response", loc="left")
    fig.tight_layout()
    return _save(fig, name)


def raw_atlas(items, name, title, per_row=6):
    """Small panels of whole recordings, each on its own vertical scale."""
    n_rows = int(np.ceil(len(items) / per_row))
    fig, axes = plt.subplots(n_rows, per_row, figsize=(19, 2 * n_rows + .7), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, it in zip(axes.ravel(), items):
        ax.set_visible(True)
        ax.plot(it["t"], it["x"], lw=.4, color=cfg.COLOR_SIGNAL)
        if it["seg"] is not None:
            ax.axvspan(*it["seg"], color="0.85", alpha=.5, lw=0)
        ax.set_title(it["title"], loc="left", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{title} (µS against time, s)", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


