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
    bottom.plot(z["t2"], z["y2"], "o", ms=4, color=cfg.COLOR_EVENTS, label=f"{item.get('fsy', 2):g} Hz samples the solver gets")
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




def _blocks(ax, t, mk, label=False):
    """Shade one input's baseline (darker) and film blocks (marker codes as saved: -1 baseline,
    1xx-3xx films); label them along the top when label is set."""
    m = np.nan_to_num(np.asarray(mk, float), nan=0.0)
    edges = np.flatnonzero(np.diff(m) != 0) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(m)]):
        code = m[a]
        if code == -1 or 100 <= code < 400:
            ax.axvspan(t[a], t[b - 1], color="0.86" if code == -1 else "0.94", lw=0, zorder=0)
            if label:
                ax.text((t[a] + t[b - 1]) / 2, 1.01, "baseline" if code == -1 else f"{int(code)}",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7,
                        color="0.35")


def _limits(*ys):
    lo = min(np.nanmin(y) for y in ys if len(y))
    hi = max(np.nanmax(y) for y in ys if len(y))
    pad = .06 * (hi - lo if hi > lo else 1)
    return lo - pad, hi + pad


def _pair_page(it, zoom_s):
    """One synthetic session against its real source: whole session, then the same zoom of both."""
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=.5, wspace=.1,
                          left=.06, right=.99, top=.86, bottom=.07)
    wr = fig.add_subplot(gs[0, :])
    ws = fig.add_subplot(gs[1, :], sharex=wr, sharey=wr)
    zr = fig.add_subplot(gs[2, 0])
    zs = fig.add_subplot(gs[2, 1], sharex=zr, sharey=zr)
    vis = it["ev_a"] >= cfg.MIN_EVENT_uS
    lo, hi = _limits(it["y_real"], it["y_syn"])
    m = np.nan_to_num(np.asarray(it["mk_real"], float))
    films = np.flatnonzero((m >= 100) & (m < 400))
    t0 = it["t_real"][films[0]] if len(films) else it["t_real"][0]
    t1 = t0 + zoom_s
    panels = ((wr, zr, it["t_real"], it["y_real"], it["mk_real"], cfg.COLOR_SIGNAL, it["real_name"]),
              (ws, zs, it["t_syn"], it["y_syn"], it["mk_syn"], cfg.COLOR_TONIC, it["syn_name"]))
    zoom_y = []
    for whole, zoom, t, y, mk, color, name in panels:
        _blocks(whole, t / 60, mk, label=whole is wr)
        whole.plot(t / 60, y, lw=.6, color=color)
        for v in (t0, t1):
            whole.axvline(v / 60, color="0.45", lw=.8, ls=":")
        whole.set_ylabel("µS")
        whole.set_title(name, loc="left", fontsize=9, pad=12 if whole is wr else 4)
        k = (t >= t0) & (t <= t1)
        zoom.plot(t[k] - t0, y[k], lw=1.2, color=color)
        zoom.set_title(f"{name}, {zoom_s:.0f} s from the first film's start (dotted lines above)",
                       loc="left", fontsize=8)
        zoom.set_xlabel("s")
        zoom_y.append(y[k])
    whole.set_xlabel("min")
    wr.set_ylim(lo, hi)
    ws.plot(it["ev_t"][vis] / 60, np.full(vis.sum(), lo + .02 * (hi - lo)), "|", ms=6,
            color=cfg.COLOR_EVENTS, label="true pulses >= 0.02 uS")
    ws.legend(loc="upper right", fontsize=7, frameon=False)
    zlo, zhi = _limits(*zoom_y)
    zr.set_ylim(zlo, zhi)
    zr.set_ylabel("µS")
    k = (it["ev_t"] >= t0) & (it["ev_t"] <= t1)
    for sel, color, label in ((k & vis, cfg.COLOR_EVENTS, "true pulses >= 0.02 uS"),
                              (k & ~vis, "0.65", "true pulses < 0.02 uS")):
        zs.plot(it["ev_t"][sel] - t0, np.full(sel.sum(), zlo + .03 * (zhi - zlo)), "|", ms=9,
                color=color, label=label)
    zs.legend(loc="upper right", fontsize=7, frameon=False)
    for ax in (wr, ws, zr, zs):
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="0.9", lw=.5)
    fig.suptitle(it["title"], fontsize=9, x=.01, ha="left")
    return fig


def _overview(items):
    """All sessions at a glance: real input left, synthetic right, same y-axis within a row."""
    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(13, .95 * n + 1.2), squeeze=False)
    for row, it in zip(axes, items):
        lo, hi = _limits(it["y_real"], it["y_syn"])
        for ax, t, y, mk, color in ((row[0], it["t_real"], it["y_real"], it["mk_real"], cfg.COLOR_SIGNAL),
                                    (row[1], it["t_syn"], it["y_syn"], it["mk_syn"], cfg.COLOR_TONIC)):
            _blocks(ax, t / 60, mk)
            ax.plot(t / 60, y, lw=.4, color=color)
            ax.set_ylim(lo, hi)
            ax.tick_params(labelsize=6)
        row[0].set_ylabel(it["label"], rotation=0, ha="right", va="center", fontsize=7)
    axes[0, 0].set_title("real input (the participant)", fontsize=9)
    axes[0, 1].set_title("synthetic input built from it (flat noise)", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("min", fontsize=7)
    fig.suptitle("Solver input (1 Hz low-pass, 4 Hz), uS; same y-axis within a row; "
                 "dark grey = baseline, light grey = films", fontsize=9)
    fig.subplots_adjust(left=.16, right=.99, top=1 - .7 / (.95 * n + 1.2), bottom=.5 / (.95 * n + 1.2),
                        hspace=.45, wspace=.08)
    return fig


def synthetic_vs_real(items, folder, pdf_name, zoom_s=120):
    """Synthetic sessions next to the real participants they were built from.
    -> results/figures/<folder>/overview.png and <name>.png, and every page in <pdf_name>."""
    from matplotlib.backends.backend_pdf import PdfPages
    out_dir = cfg.FIG_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(cfg.FIG_DIR / pdf_name) as pdf:
        for name, fig in [("overview", _overview(items))] + [(it["name"], None) for it in items]:
            fig = fig or _pair_page(next(i for i in items if i["name"] == name), zoom_s)
            pdf.savefig(fig)
            fig.savefig(out_dir / f"{name}.png", dpi=110)
            plt.close(fig)
    print(f"saved {cfg.FIG_DIR / pdf_name} and {len(items) + 1} PNGs in {out_dir}")


COLOR_SIGNAL, COLOR_TONIC, COLOR_EVENTS = cfg.COLOR_SIGNAL, cfg.COLOR_TONIC, cfg.COLOR_EVENTS
COLOR_MISSED = "#c8433a"     # a true pulse the solver did not find
COLOR_FALSE = "#7a4fc4"      # an estimated pulse with no true one within 1 s


def _pulse_rows(ax, rows, lo, span):
    """Tick rows under the signal: [(label, times, color)], first row lowest."""
    for k, (label, times, color) in enumerate(rows):
        ax.plot(times, np.full(len(times), lo - (0.04 + 0.05 * k) * span), "|", ms=7, mew=1.2,
                color=color, label=f"{label} ({len(times)})")


def _deconv_page(it, zoom_s):
    """One session: input, reconstruction and tonic, whole and zoomed, with its pulses."""
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=.35, left=.06, right=.99, top=.88, bottom=.07)
    whole, zoom = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    t, m = it["t"], np.nan_to_num(np.asarray(it["mk"], float))
    films = np.flatnonzero((m >= 100) & (m < 400))
    t0 = t[films[0]] if len(films) else t[0]
    for ax, sl, scale, xlabel in ((whole, slice(None), 60, "min"),
                                  (zoom, (t >= t0) & (t <= t0 + zoom_s), 1, "s from the first film's start")):
        tt = t[sl] if not isinstance(sl, slice) else t
        off = 0 if scale == 60 else t0
        y, rec, ton = (np.asarray(it[k])[sl] for k in ("y", "y_rec", "tonic"))
        lo, hi = np.nanmin(y), np.nanmax(y)
        span = hi - lo if hi > lo else 1
        if scale == 60:
            _blocks(ax, tt / 60, np.asarray(it["mk"])[sl] if not isinstance(sl, slice) else it["mk"], label=True)
        ax.plot((tt - off) / scale, y, lw=.6 if scale == 60 else 1.0, color=COLOR_SIGNAL, label="input (before)")
        ax.plot((tt - off) / scale, rec, lw=.6 if scale == 60 else 1.2, color=COLOR_TONIC,
                label="reconstruction (after)")
        ax.plot((tt - off) / scale, ton, lw=.8, color="0.35", ls="--", label="tonic")
        rows = []
        for label, times, color in it["pulses"]:
            times = np.asarray(times)
            keep = (times >= tt[0]) & (times <= tt[-1])
            rows.append((label, (times[keep] - off) / scale, color))
        _pulse_rows(ax, rows, lo, span)
        ax.set_ylim(lo - (0.06 + 0.05 * len(rows)) * span, hi + .05 * span)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("µS")
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="0.9", lw=.5)
    whole.legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
    zoom.set_title(f"{zoom_s:.0f} s from the first film's start", loc="left", fontsize=8)
    fig.suptitle(it["title"], fontsize=9, x=.01, ha="left")
    return fig


def deconv_pages(items, folder, pdf_name, zoom_s=120):
    """Signals before and after deconvolution, with their pulses.
    -> results/figures/<folder>/<name>.png and every page in <pdf_name>."""
    from matplotlib.backends.backend_pdf import PdfPages
    out_dir = cfg.FIG_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(cfg.FIG_DIR / pdf_name) as pdf:
        for it in items:
            fig = _deconv_page(it, zoom_s)
            pdf.savefig(fig)
            fig.savefig(out_dir / f"{it['name']}.png", dpi=110)
            plt.close(fig)
    print(f"saved {cfg.FIG_DIR / pdf_name} and {len(items)} PNGs in {out_dir}")


STAGE2_MIN_uS = 0.05         # stage 2 counts pulses at least this high (AGENTS.md, rule 8)
FIT_ROWS = ("1  raw recording (1000 Hz, every 10th sample drawn)",
            "2  solver input (1 Hz low-pass, 4 Hz; offset by a constant)",
            "3  output: pulses, height of each response (µS)",
            "4  reconstruction (tonic + responses) over the input")


def _fit_page(it, zoom_s):
    """One real participant, four rows (FIT_ROWS); the whole analysed segment on the left, zoom_s
    from the first film's start on the right."""
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[2.6, 1], hspace=.5, wspace=.12,
                          left=.06, right=.99, top=.9, bottom=.06)
    t, mk = it["t"], np.nan_to_num(np.asarray(it["mk"], float))
    films = np.flatnonzero((mk >= 100) & (mk < 400))
    t0 = t[films[0]] if len(films) else t[0]
    h, big = it["ev_h"], it["ev_h"] >= STAGE2_MIN_uS
    hmax = 1.08 * (h.max() if len(h) else 0.1)
    for col, (a, b, scale, off) in enumerate(((t[0], t[-1], 60, 0.0), (t0, t0 + zoom_s, 1, t0))):
        x = lambda v: (np.asarray(v) - off) / scale
        top = fig.add_subplot(gs[0, col])
        axes = [top] + [fig.add_subplot(gs[r, col], sharex=top) for r in range(1, 4)]
        m = (it["raw_t"] >= a) & (it["raw_t"] <= b)
        s = (t >= a) & (t <= b)
        e = (it["ev_t"] >= a) & (it["ev_t"] <= b)
        axes[0].plot(x(it["raw_t"][m]), it["raw_x"][m], lw=.5, color="0.2", rasterized=True)
        axes[1].plot(x(t[s]), it["y"][s], lw=.7, color=COLOR_SIGNAL)
        axes[2].vlines(x(it["ev_t"][e & big]), 0, h[e & big], color=COLOR_EVENTS, lw=1,
                       label=f"≥ {STAGE2_MIN_uS:g} µS, main analysis ({int(big.sum())})")
        axes[2].vlines(x(it["ev_t"][e & ~big]), 0, h[e & ~big], color="0.6", lw=1,
                       label=f"< {STAGE2_MIN_uS:g} µS ({int((~big).sum())})")
        axes[2].axhline(STAGE2_MIN_uS, color="0.4", ls=":", lw=.8)
        axes[2].set_ylim(0, hmax)
        axes[3].plot(x(t[s]), it["y"][s], lw=.7, color=COLOR_SIGNAL, label="input")
        axes[3].plot(x(t[s]), it["y_rec"][s], lw=.9, color=COLOR_TONIC, label="reconstruction")
        axes[3].plot(x(t[s]), it["tonic"][s], lw=.8, color="0.35", ls="--", label="tonic")
        for r, ax in enumerate(axes):
            if col == 0:
                _blocks(ax, t / 60, it["mk"], label=r == 0)
                ax.set_title(FIT_ROWS[r], loc="left", fontsize=8, pad=12 if r == 0 else 4)
            ax.set_ylabel("µS", fontsize=7)
            ax.tick_params(labelsize=7, labelbottom=r == 3)
            ax.grid(axis="y", color="0.9", lw=.5)
        axes[0].set_xlim(x(a), x(b))
        axes[3].set_xlabel("min" if col == 0 else "s from the first film's start", fontsize=8)
        if col == 1:
            axes[0].set_title(f"zoom: {zoom_s:.0f} s from the first film's start", loc="left", fontsize=8, pad=12)
        else:
            axes[2].legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
            axes[3].legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
    fig.suptitle(it["title"], fontsize=9, x=.01, ha="left")
    return fig


def fit_pages(items, pdf_name, zoom_s=120):
    """Real participants' fits to judge by eye, one page each (_fit_page).  items may be a generator,
    so only one participant's raw recording is in memory at a time.  -> results/figures/<pdf_name>"""
    from matplotlib.backends.backend_pdf import PdfPages
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with PdfPages(cfg.FIG_DIR / pdf_name) as pdf:
        for it in items:
            fig = _fit_page(it, zoom_s)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            n += 1
    print(f"saved {cfg.FIG_DIR / pdf_name} ({n} pages)")


COLOR_ZOOM = "#e9c46a"       # the stretch shown enlarged on the right


def _diff_window(rows, zoom_s):
    """Start and end (s) of the zoom_s stretch where the rows' tonics differ most from the first
    row's (the control): the moving mean of the largest absolute difference, at its maximum."""
    t = np.asarray(rows[0]["t"], float)
    base = np.asarray(rows[0]["tonic"], float)
    d = np.zeros(len(t))
    for r in rows[1:]:
        d = np.maximum(d, np.abs(np.interp(t, np.asarray(r["t"], float), np.asarray(r["tonic"], float)) - base))
    n = max(int(zoom_s / float(np.median(np.diff(t)))), 1)
    if len(d) <= n:
        return t[0], t[-1]
    a = int(np.argmax(np.convolve(d, np.ones(n) / n, mode="valid")))
    return t[a], t[a + n - 1]


def tonic_compare_pages(pages, folder, pdf_name, zoom_s=120):
    """One page per session, one row per setting (the first is the control): input, reconstruction
    and tonic over the whole session (left) and over the zoom_s stretch where the tonics differ
    most (right), with found / missed / false pulses under each.
    pages: [{"name", "title", "rows": [runs._deconv_item dicts, each with its own title]}].
    -> results/figures/<folder>/<name>.png and every page in <pdf_name>."""
    from matplotlib.backends.backend_pdf import PdfPages
    out_dir = cfg.FIG_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(cfg.FIG_DIR / pdf_name) as pdf:
        for pg in pages:
            rows = pg["rows"]
            a, b = _diff_window(rows, zoom_s)
            height = 2.9 * len(rows) + 0.8
            fig = plt.figure(figsize=(14, height))
            gs = fig.add_gridspec(len(rows), 2, width_ratios=[2.2, 1], hspace=.6, wspace=.1,
                                  left=.05, right=.99, top=1 - .75 / height, bottom=.5 / height)
            for k, it in enumerate(rows):
                t = np.asarray(it["t"], float)
                y, rec, ton = (np.asarray(it[col], float) for col in ("y", "y_rec", "tonic"))
                mk = np.asarray(it["mk"])
                for j, (keep, scale, off) in enumerate(((np.ones(len(t), bool), 60, 0.0),
                                                        ((t >= a) & (t <= b), 1, a))):
                    ax = fig.add_subplot(gs[k, j])
                    tt, yy, rr, oo = t[keep], y[keep], rec[keep], ton[keep]
                    lo, hi = np.nanmin(yy), np.nanmax(yy)
                    span = hi - lo if hi > lo else 1
                    x = (tt - off) / scale
                    if j == 0:
                        _blocks(ax, x, mk[keep])
                        ax.axvspan(a / 60, b / 60, color=COLOR_ZOOM, alpha=.35, lw=0, zorder=0)
                    ax.plot(x, yy, lw=.6 if j == 0 else 1.0, color=COLOR_SIGNAL, label="input")
                    ax.plot(x, rr, lw=.6 if j == 0 else 1.2, color=COLOR_TONIC, label="reconstruction")
                    ax.plot(x, oo, lw=.9, color="0.35", ls="--", label="tonic")
                    if "true_tonic" in it:
                        ax.plot(x, np.asarray(it["true_tonic"], float)[keep], lw=1.0, color="k", ls=":",
                                label="true tonic")
                    ticks = []
                    for label, times, color in it["pulses"]:
                        times = np.asarray(times, float)
                        inside = (times >= tt[0]) & (times <= tt[-1])
                        ticks.append((label, (times[inside] - off) / scale, color))
                    _pulse_rows(ax, ticks, lo, span)
                    ax.set_ylim(lo - (0.06 + 0.05 * len(ticks)) * span, hi + .05 * span)
                    ax.tick_params(labelsize=7)
                    ax.grid(axis="y", color="0.9", lw=.5)
                    if j == 0:
                        ax.set_title(it["title"], loc="left", fontsize=8)
                        ax.set_ylabel("µS")
                        ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=3)
                    else:
                        ax.set_title(f"{a / 60:.1f}-{b / 60:.1f} min: where the tonics differ most",
                                     loc="left", fontsize=7)
                    if k == len(rows) - 1:
                        ax.set_xlabel("min" if j == 0 else f"s from {a / 60:.1f} min")
            fig.suptitle(pg["title"], fontsize=9, x=.01, ha="left")
            pdf.savefig(fig)
            fig.savefig(out_dir / f"{pg['name']}.png", dpi=100)
            plt.close(fig)
    print(f"saved {cfg.FIG_DIR / pdf_name} and {len(pages)} PNGs in {out_dir}")
