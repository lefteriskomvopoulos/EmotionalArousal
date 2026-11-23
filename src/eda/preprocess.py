"""
The input of our own deconvolution: one function, one order, shown step by step.

prepare() is the only place the order lives, and both the input files and the figures come from
it, so the picture is always what the solver receives.  All steps run at the signal's original
rate, as Amin & Faghih (IEEE Access 2019) state: "We perform all the preprocessing in the original
sampling frequency", with the single down-sampling last.

  1 raw            the recording, 1000 Hz
  2 fill gaps      linear interpolation, because the steps below need a gap-free signal
  3 jumps          peaks of prominence 0.1 on the change over 10 ms; 0.5 s centred on each is
                   replaced by a cubic B-spline
  4 low-pass       FIR, Hamming, 6.4 s long, 0.5 Hz, no delay
  5 cut            first baseline sample to last film sample, after the filter
  6 down-sample    to 2 Hz: the only rate change.  Step 4 leaves nothing above 1 Hz to fold back
  7 shift          a constant, only if the 2 Hz signal goes below 0

Why this order, against the first version (paper_preprocess, black-box branch):
  - the first version dropped to 100 Hz with no filter before everything else, so content above
    50 Hz folded back and jumps were looked for on a thinned signal.  Now there is no such step.
  - it cut the segment before filtering, so the filter's edges fell on the data and were padded
    with end values.  Now they fall on recording outside the segment.
  - the filter's length and the jump span are set in seconds, so they mean the same at any rate:
    at 100 Hz this is exactly the first version's order 640 and one-sample difference.

Tonic removal for 'faghih2014' is not a step yet.  In the 2019 paper it comes after the low-pass
and before resampling.

Input variants (VARIANTS): the same chain with another low-pass cutoff and solver rate.
  own    0.5 Hz, 2 Hz   Amin & Faghih 2021
  own4   1 Hz, 4 Hz     tuning round 3: the 0.5 Hz filter smears each response by about a
                        second, which the fit may read as a longer tau_rise
A variant's synthetic noise is matched to the real participants run through that same variant,
so the solver sees as much noise as it would on real data (data/input_check_<variant>.csv).

Also here, for the inputs and their checks: typical_size / input_scale (an input's typical
response size, model-free), the raw atlas and the stage figures.  The synthetic examples in those
two figures are the first generator's sessions (synth_v1).

Saved: data/preprocess_stages.csv, data/raw_overview.csv, data/input_check.csv,
       data/filter_response.csv, results/figures/filter_response.png,
       results/figures/stages_<name>.png, results/figures/raw_atlas_<group>.png
"""
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.io import loadmat
from scipy.signal import fftconvolve, find_peaks, firwin

from . import config as cfg
from . import figures
from . import tbme_export as tx
from .dataset import baseline_mask, film_mask, load

REAL_IDS = (53, 115, 26, 3, 80)                  # the 5 real participants shown and exported

STAGES_CSV = cfg.ROOT / "data" / "preprocess_stages.csv"
OVERVIEW_CSV = cfg.ROOT / "data" / "raw_overview.csv"
CHECK_CSV = cfg.ROOT / "data" / "input_check.csv"
FILTER_CSV = cfg.ROOT / "data" / "filter_response.csv"
RAW_FS = 1000                                    # Hz, the recordings
FSY = tx.FSY                                     # Hz, the solver's signal rate
JUMP_PROMINENCE = 0.1                            # uS (paper)
JUMP_SPAN_S = 0.01                               # one sample at the paper's 100 Hz
JUMP_PATCH_S = 0.5                               # s (paper)
ANCHOR_S, ANCHOR_SIDE_S = 0.1, 1.0               # spline anchors: 0.1 s means, 1 s each side
FIR_SECONDS = 6.4                                # the first version's order 640 at 100 Hz
FIR_CUTOFF_HZ = 0.5                              # Hz (paper)
VARIANTS = {"own": (FIR_CUTOFF_HZ, FSY),         # suffix: (low-pass cutoff Hz, solver rate Hz)
            "own4": (1.0, 4)}
ZOOM_S = 60
ATLAS_HZ = 10                                    # thumbnails only
SCALE_MIN_PEAKS = 5      # fewer baseline peaks than this: the whole input is used


def find_jumps(x, fs):
    """Centres of the jumps: peaks of the change over JUMP_SPAN_S."""
    k = max(int(round(JUMP_SPAN_S * fs)), 1)
    return find_peaks(np.abs(x[k:] - x[:-k]), prominence=JUMP_PROMINENCE)[0] + (k + 1) // 2


def remove_jumps(x, t, fs):
    """0.5 s around each jump -> cubic B-spline through 0.1 s means on either side.

    Means, not samples: a spline through raw samples follows their noise and quantisation, and
    can swing far off across a 0.5 s hole.
    """
    centres = find_jumps(x, fs)
    if not len(centres):
        return x, 0
    half = int(JUMP_PATCH_S * fs / 2)
    drop = np.zeros(len(x), bool)
    for c in centres:
        drop[max(c - half, 0):c + half] = True
    edges = np.flatnonzero(np.diff(np.r_[0, drop.astype(int), 0]))
    out = x.copy()
    n_anchor, block = int(ANCHOR_SIDE_S / ANCHOR_S), int(ANCHOR_S * fs)
    for a, b in zip(edges[::2], edges[1::2]):
        idx = [(a - (k + 1) * block, a - k * block) for k in reversed(range(n_anchor))] \
            + [(b + k * block, b + (k + 1) * block) for k in range(n_anchor)]
        idx = [(p, q) for p, q in idx if p >= 0 and q <= len(x) and not drop[p:q].any()]
        if not idx:
            continue
        ta = np.array([t[p:q].mean() for p, q in idx])
        xa = np.array([x[p:q].mean() for p, q in idx])
        both_sides = len(ta) >= 4 and ta[0] < t[a] and ta[-1] > t[b - 1]    # a spline must not extrapolate
        out[a:b] = make_interp_spline(ta, xa, k=3)(t[a:b]) if both_sides else np.interp(t[a:b], ta, xa)
    return out, len(centres)


def low_pass(x, fs, cutoff=FIR_CUTOFF_HZ):
    """Zero-delay FIR low-pass, the same in seconds and Hz at any rate; ends padded by end values."""
    order = int(FIR_SECONDS * fs)
    b = firwin(order + 1, cutoff, fs=fs, window="hamming")
    d = order // 2
    return fftconvolve(np.r_[np.full(d, x[0]), x, np.full(d, x[-1])], b, mode="valid")


def prepare(x, t, fs, keep, cutoff=FIR_CUTOFF_HZ, fsy=FSY):
    """Steps 2-7.  keep: the indices of x to cut to, before taking every fs/fsy-th -> all stages."""
    filled, gaps = tx.fill(x, t)
    clean, n_jumps = remove_jumps(filled, t, fs)
    low = low_pass(clean, fs, cutoff)
    idx = np.arange(keep[0], keep[-1] + 1, int(fs // fsy))
    y = low[idx]
    shift = 0.0
    if y.min() < 0:
        y, shift = tx.set_floor(y, 0.0)
    return {"filled": filled, "gaps": gaps, "clean": clean, "n_jumps": n_jumps, "low": low,
            "idx": idx, "t2": t[idx], "y": y, "shift": shift}


def segment_of(marker):
    """First baseline sample to last film sample, at the signal's own rate."""
    return np.flatnonzero(baseline_mask(marker))[0], np.flatnonzero(film_mask(marker))[-1]


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def noise_2hz(y):
    """Noise of the input in quiet stretches: robust size of its second difference, at the
    input's own rate (2 Hz for 'own', 4 Hz for 'own4')."""
    return 1.4826 * float(np.median(np.abs(np.diff(y, 2)))) / np.sqrt(6)


def filter_response():
    """What the low-pass really does: gain against frequency, the same at any rate."""
    b = firwin(int(FIR_SECONDS * cfg.FS) + 1, FIR_CUTOFF_HZ, fs=cfg.FS, window="hamming")
    n = 2 ** 16
    f, gain = np.fft.rfftfreq(n, 1 / cfg.FS), np.abs(np.fft.rfft(b, n))
    out = pd.DataFrame({"f_hz": f, "gain": gain}).query("f_hz <= 3")
    out.to_csv(FILTER_CSV, index=False)
    figures.filter_response(out, FIR_CUTOFF_HZ, FSY / 2, "filter_response.png")
    print("filter gain at " + ", ".join(f"{v} Hz: {np.interp(v, f, gain):.3f}" for v in (0.1, 0.25, 0.5, 0.75, 1)))


def check_inputs():
    """All 119 through prepare(): input roughness, and jumps and gaps inside the kept segment."""
    q = pd.read_csv(cfg.QUALITY_CSV)
    q = q[q.usable].sort_values("id")
    rows = []
    for k, r in enumerate(q.itertuples(), 1):
        d = load(int(r.id), every=1)
        i0, i1 = segment_of(d["marker"])
        p = prepare(d["eda"], d["t"], RAW_FS, (i0, i1))
        c = find_jumps(p["filled"], RAW_FS)
        rows.append({"id": int(r.id), "group": r.group, "noise_2hz_uS": noise_2hz(p["y"]), "jumps_all": len(c),
                     "jumps_kept": int(((c >= i0) & (c <= i1)).sum()), "gaps_kept": int(p["gaps"][i0:i1 + 1].sum()),
                     "shift_uS": p["shift"], "input_min_uS": p["y"].min(), "input_max_uS": p["y"].max()})
        if k % 20 == 0:
            print(f"  {k}/{len(q)} done")
    out = pd.DataFrame(rows)
    out.to_csv(CHECK_CSV, index=False)
    filter_response()
    print(out[["noise_2hz_uS", "jumps_all", "jumps_kept", "gaps_kept", "shift_uS"]].describe()
          .loc[["min", "50%", "max"]].to_string(float_format=lambda v: f"{v:.4g}"))
    print("participants with a jump inside the kept segment:", int((out.jumps_kept > 0).sum()), "of", len(out))
    print("saved", CHECK_CSV.name, FILTER_CSV.name)
    return out


def picture(name, x, t, fs, p, first_row, old, cutoff=FIR_CUTOFF_HZ, fsy=FSY):
    """The figure's rows and the table's rows for one session."""
    k, i0, i1 = int(5 * fs), p["idx"][0], p["idx"][-1]
    kept = p["low"][i0:i1]
    centre = i0 + np.argmax(kept[k:][::10] - kept[:-k][::10]) * 10 + k // 2     # steepest rise in the segment
    a, b = max(centre - ZOOM_S * fs // 2, 0), min(centre + ZOOM_S * fs // 2, len(t))
    keep = (p["t2"] >= t[a]) & (p["t2"] < t[b - 1])
    jump, low = p["clean"] - p["filled"], p["low"] - p["clean"]
    rows = [first_row,
            {"label": "2 gaps filled", "t": t, "x": p["filled"], "change": None,
             "note": f"{int(p['gaps'].sum())} samples filled ({100 * p['gaps'].mean():.2f}%)"},
            {"label": "3 jumps removed", "t": t, "x": p["clean"], "change": jump,
             "note": f"{p['n_jumps']} jumps, {int((jump != 0).sum())} samples replaced"},
            {"label": f"4 low-pass {cutoff:g} Hz", "t": t, "x": p["low"], "change": low,
             "note": f"removed: rms {rms(low):.4f} µS, largest {np.abs(low).max():.3f} µS"},
            {"label": f"5-7 cut, {fsy:g} Hz, shift", "t": p["t2"], "x": p["y"], "t_change": p["t2"],
             "change": None if old is None else p["y"] - old,
             "note": f"{len(p['y'])} samples, shift +{p['shift']:.3f} µS"
                     + ("" if old is None else f"; this minus the first version: rms {rms(p['y'] - old):.4f} µS")}]
    z = {"t": t[a:b], "before": p["clean"][a:b], "low": p["low"][a:b], "t2": p["t2"][keep],
         "y2": p["y"][keep] - p["shift"]}
    table = {"session": name, "rate_hz": fs, "minutes_recorded": len(x) / fs / 60,
             "minutes_kept": len(p["y"]) / fsy / 60, "margin_before_s": p["t2"][0] - t[0],
             "margin_after_s": t[-1] - p["t2"][-1], "gap_samples": int(p["gaps"].sum()), "jumps": p["n_jumps"],
             "lowpass_rms_uS": rms(low), "lowpass_max_uS": float(np.abs(low).max()), "shift_uS": p["shift"],
             "input_min_uS": p["y"].min(), "input_max_uS": p["y"].max(), "noise_2hz_uS": noise_2hz(p["y"]),
             "vs_first_rms_uS": np.nan if old is None else rms(p["y"] - old),
             "vs_first_max_uS": np.nan if old is None else float(np.abs(p["y"] - old).max())}
    return {"name": name, "rows": rows, "zoom": z}, table


def real_session(pid):
    d = load(pid, every=1)
    x, t = d["eda"], d["t"]
    i0, i1 = segment_of(d["marker"])
    p = prepare(x, t, RAW_FS, (i0, i1))
    first = {"label": "1 raw 1000 Hz", "t": t, "x": x, "change": None, "seg": (t[i0], t[i1]),
             "note": f"{len(x) / RAW_FS / 60:.1f} min, grey = kept ({(i1 - i0) / RAW_FS / 60:.1f} min)"}
    first_version = tx.OUT / f"P{pid}_pp.mat"    # written on the black-box branch (paper-export)
    old = None
    if first_version.exists():                   # compared only where that file is present
        m = loadmat(first_version)
        old = np.interp(p["t2"], m["t"].ravel(), m["y"].ravel())   # the two 2 Hz grids differ slightly
    return picture(f"P{pid}", x, t, RAW_FS, p, first, old)


def stages(real_ids=REAL_IDS):
    """Every step, whole timeline: 5 real participants and 5 tuning sessions."""
    from .noise import matched
    from .synth_v1 import tuning_sessions
    table = []
    for pid in real_ids:
        item, row = real_session(pid)
        table.append(row)
        figures.preprocess_stages(item, f"stages_{item['name']}.png")
    targets = pd.read_csv(CHECK_CSV).set_index("id").noise_2hz_uS.to_dict()
    for name, row, g, tr, td, t, mk, s in tuning_sessions((("prior", 2), ("moderate", 1), ("far", 2))):
        x = matched(name, row, s, t, targets)
        p = prepare(x, t, cfg.FS, (0, len(t) - 1))
        first = {"label": "1 generated 100 Hz", "t": t, "x": x, "change": None,
                 "note": f"τ group {g}, true τ {tr:.2f}/{td:.2f} s, {len(s['ev_t'])} true impulses"}
        item, trow = picture(name, x, t, cfg.FS, p, first, None)
        table.append(trow)
        figures.preprocess_stages(item, f"stages_{name}.png")
    out = pd.DataFrame(table)
    out.to_csv(STAGES_CSV, index=False)
    print(out[["session", "margin_before_s", "gap_samples", "jumps", "lowpass_rms_uS", "shift_uS",
               "noise_2hz_uS", "vs_first_rms_uS", "vs_first_max_uS"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("saved", STAGES_CSV.name)
    return out


def input_scale(name):
    """Typical response size of one solver input: the median prominence of its peaks >= 0.02 uS
    in the baseline (stage 2's base_scr_med_uS), or over the whole input when the baseline has
    fewer than SCALE_MIN_PEAKS.  Model-free, and the baseline keeps it apart from the film responses."""
    m = loadmat(tx.OUT / f"{name}.mat")
    return typical_size(m["y"].ravel(), m["marker"].ravel())


def typical_size(y, mk):
    """input_scale on arrays: a solver input y and its numeric block codes mk (-1 = baseline)."""
    for where, x in (("baseline", y[mk == -1]), ("whole", y)):
        pr = find_peaks(x, prominence=cfg.MIN_EVENT_uS)[1]["prominences"]
        if len(pr) >= SCALE_MIN_PEAKS:
            break
    return {"scale_uS": float(np.median(pr)) if len(pr) else np.nan, "n_peaks": len(pr), "from": where}


def overview_row(pid, group, d):
    x, t = d["eda"], d["t"]
    i0, i1 = segment_of(d["marker"])
    ok = x[~np.isnan(x)]
    kept = x[i0:i1 + 1]
    return {"id": pid, "group": group, "minutes": (t[-1] - t[0]) / 60, "t_first_s": t[0],
            "first_uS": ok[0], "min_uS": ok.min(), "max_uS": ok.max(), "share_negative": (ok < 0).mean(),
            "share_missing": np.isnan(x).mean(), "kept_from_s": t[i0], "kept_to_s": t[i1],
            "margin_before_s": t[i0] - t[0], "margin_after_s": t[-1] - t[i1],
            "kept_min_uS": np.nanmin(kept), "kept_max_uS": np.nanmax(kept)}


def atlas():
    """All 119 usable participants and the 18 tuning sessions, before any processing."""
    from .synth_v1 import tuning_sessions
    q = pd.read_csv(cfg.QUALITY_CSV)
    q = q[q.usable].sort_values(["group", "id"])
    every = int(cfg.FS // ATLAS_HZ)
    rows, items = [], {}
    for k, r in enumerate(q.itertuples(), 1):
        d = load(int(r.id))
        rows.append(overview_row(int(r.id), r.group, d))
        i0, i1 = segment_of(d["marker"])
        items.setdefault(r.group, []).append({"title": f"P{r.id}", "t": d["t"][::every], "x": d["eda"][::every],
                                              "seg": (d["t"][i0], d["t"][i1])})
        if k % 20 == 0:
            print(f"  {k}/{len(q)} read")
    out = pd.DataFrame(rows)
    out.to_csv(OVERVIEW_CSV, index=False)
    for group, its in items.items():
        figures.raw_atlas(its, f"raw_atlas_{group}.png", f"{group}: raw skin conductance, grey = kept segment")
    its = [{"title": f"{name} {g}", "t": t[::every], "x": s["x"][::every], "seg": None}
           for name, row, g, tr, td, t, mk, s in tuning_sessions()]
    figures.raw_atlas(its, "raw_atlas_synth_tune.png", "18 tuning sessions before preprocessing")
    print(out[["minutes", "first_uS", "min_uS", "max_uS", "share_negative", "share_missing", "margin_before_s"]]
          .describe().loc[["min", "50%", "max"]].to_string(float_format=lambda v: f"{v:.3g}"))
    print("below 0 uS somewhere:", int((out.min_uS < 0).sum()), "of", len(out))
    print("saved", OVERVIEW_CSV.name)
    return out


def stages_variant(pid=53, suffix="own4"):
    """preprocess-stages-own4: the stage figure of one real participant through a variant's chain
    (own4: 1 Hz low-pass, 4 Hz, the solver's input from round 3 on).  Written next to the original
    figures under its own name, so those stay as they are.
    -> results/figures/stages_P<pid>_<suffix>.png"""
    cutoff, fsy = VARIANTS[suffix]
    d = load(pid, every=1)
    x, t = d["eda"], d["t"]
    i0, i1 = segment_of(d["marker"])
    p = prepare(x, t, RAW_FS, (i0, i1), cutoff, fsy)
    first = {"label": "1 raw 1000 Hz", "t": t, "x": x, "change": None, "seg": (t[i0], t[i1]),
             "note": f"{len(x) / RAW_FS / 60:.1f} min, grey = kept ({(i1 - i0) / RAW_FS / 60:.1f} min)"}
    item, _ = picture(f"P{pid}", x, t, RAW_FS, p, first, None, cutoff, fsy)
    item["fsy"] = fsy
    figures.preprocess_stages(item, f"stages_P{pid}_{suffix}.png")
