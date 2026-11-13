"""
One participant selection for the real-data analysis: 187 recordings -> the 50 deconvolved.

  stage 1  the first screen (quality.py, fixed before the proposal): 187 -> 119 usable.
           Signals that are physiologically unsuitable: the dataset authors' EDA flag, electrode
           noise, missing data, near-flat recordings, electrodermal non-responders.
  stage 2  signals our method cannot read reliably, from what the synthetic tests showed
           (pulses of 0.02-0.05 uS sit on the noise floor; real noise is slower than flat).
           Measured on the BASELINE only, because film-period SCR density is the outcome of
           H1-H3: selecting on it would build the result into the selection.
  stage 3  a seeded random draw within each group among those who pass: QUOTA below
           (20 neutral, 10 positive, 10 negative, 10 mixed), covering H1-H6.

All measures are model-free and taken on the solver's own input (own4: 1 Hz low-pass, 4 Hz),
the same for every group.  The stage-2 rule was set once from the printed distributions
(select-measures) and written below before select was run.

Stage-2 measures, baseline (5 min) only:
  base_scr_pm       peaks of prominence >= 0.02 uS per minute
  base_scr_med_uS   their median prominence
  base_noise_uS     robust roughness of the input (preprocess.noise_2hz)
  base_snr          base_scr_med_uS / base_noise_uS: how far a typical response stands above noise
  base_slow_uS2     power at 0.06-0.3 Hz in quiet 16 s stretches (no counted response nearby);
                    NaN when the baseline has no quiet stretch
  jumps_base, jumps_segment   jumps found (and repaired) in the baseline / the analysed segment.
                    Jumps are technical, not SCR density, but movement may differ between film
                    groups, so a segment-wide rule is checked per group before it is used.

Saved: data/selection_measures.csv (select-measures), data/selection.csv and
data/selection_balance.csv (select).
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kruskal

from . import config as cfg
from . import noise as nz
from . import preprocess as pp
from .dataset import baseline_mask, load

MEASURES_CSV = cfg.ROOT / "data" / "selection_measures.csv"
SELECTION_CSV = cfg.ROOT / "data" / "selection.csv"
BALANCE_CSV = cfg.ROOT / "data" / "selection_balance.csv"
QUOTA = {"neutral": 20, "positive": 10, "negative": 10, "mixed": 10}
SELECT_SEED = 2050
VARIANT = "own4"

# Stage-2 rule, agreed 2026-11-10 from select-measures (119 stage-1 survivors), before select ran:
#   median baseline SCR >= 0.05 uS.  Low-amplitude responders: below it most of a participant's
#   responses lie in 0.02-0.05 uS, where the synthetic validation recovers 58% at 65% precision,
#   against 98% / 96% above (own-errors).  0.05 uS is 5 resolution steps and the top of the
#   0.01-0.05 uS minimum-SCR range in the literature (Boucsein et al. 2012); stage 1's
#   non-responder rule is the same idea at 0.02 uS.  Excludes 22 (10 negative, 5 neutral,
#   4 positive, 3 mixed); the baseline precedes every film, so the film group cannot cause it.
# Measured but not used:
#   jumps_segment / jumps_base  zero in all 119 (stage 1's electrode-noise rule removed them)
#   base_slow_uS2               no quiet baseline stretch in 48 of 119: cannot be judged
#   base_snr, base_scr_pm       high for nearly all (SNR median 48); fast roughness is not the
#                               slow noise that hurts small pulses, and near-flat recordings are
#                               stage 1's rule
MIN_BASE_SCR_MED_uS = 0.05

STAGE2 = ["base_scr_pm", "base_scr_med_uS", "base_noise_uS", "base_snr", "base_slow_uS2",
          "jumps_base", "jumps_segment"]


def baseline_measures(pid):
    """Stage-2 measures of one participant, from the solver's input restricted to the baseline."""
    cutoff, fsy = pp.VARIANTS[VARIANT]
    d = load(pid, every=1)
    i0, i1 = pp.segment_of(d["marker"])
    p = pp.prepare(d["eda"], d["t"], pp.RAW_FS, (i0, i1), cutoff, fsy)
    base = baseline_mask(d["marker"])
    yb = p["y"][base[p["idx"]]]
    minutes = len(yb) / fsy / 60
    peaks, prop = find_peaks(yb, prominence=cfg.MIN_EVENT_uS)
    med = float(np.median(prop["prominences"])) if len(peaks) else 0.0
    noise = pp.noise_2hz(yb)
    sp, n_quiet = nz.spectra(yb, fsy)
    f, pq = sp["quiet"]
    slow = nz.band_power(f, pq, 0.06, 0.3) if n_quiet else np.nan
    jumps = pp.find_jumps(p["filled"], pp.RAW_FS)
    bi = np.flatnonzero(base)
    return {"base_min": minutes, "base_level_uS": float(np.median(yb - p["shift"])),
            "base_scr_pm": len(peaks) / max(minutes, 1e-9), "base_scr_med_uS": med,
            "base_noise_uS": noise, "base_snr": med / noise if noise > 0 else np.nan,
            "base_slow_uS2": slow, "base_quiet_windows": n_quiet,
            "jumps_base": int(((jumps >= bi[0]) & (jumps <= bi[-1])).sum()) if len(bi) else 0,
            "jumps_segment": int(((jumps >= i0) & (jumps <= i1)).sum())}


def measures():
    """select-measures: every usable participant's stage-2 measures, and their spread per group."""
    q = pd.read_csv(cfg.QUALITY_CSV)
    q = q[q.usable].sort_values("id")
    rows = []
    for k, r in enumerate(q.itertuples(), 1):
        rows.append({"id": int(r.id), "group": r.group, **baseline_measures(int(r.id))})
        if k % 20 == 0:
            print(f"  {k}/{len(q)} done")
    out = pd.DataFrame(rows)
    out.to_csv(MEASURES_CSV, index=False)
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for col in STAGE2:
        t = out.groupby("group")[col].quantile(qs).unstack()
        t.loc["all"] = out[col].quantile(qs).to_numpy()
        t.columns = [f"q{int(v * 100)}" for v in qs]
        print(f"\n{col} (stage 1 survivors, {len(out)})")
        print(t.to_string(float_format=lambda v: f"{v:.4g}"))
    print("\nquiet baseline stretches: none in", int((out.base_quiet_windows == 0).sum()), "of", len(out))
    print("saved", MEASURES_CSV.name)
    return out


def stage2_reason(r):
    """Why a stage-1 survivor fails stage 2 ('' when it passes)."""
    return "low-amplitude responder" if r.base_scr_med_uS < MIN_BASE_SCR_MED_uS else ""


def select():
    """select: stage 2 on the stage-1 survivors, then the seeded draw within each group."""
    q = pd.read_csv(cfg.QUALITY_CSV)[["id", "group", "usable"]]
    m = pd.read_csv(MEASURES_CSV)
    out = q.merge(m.drop(columns="group"), on="id", how="left")
    out["stage2_reason"] = ""
    s1 = out.usable
    out.loc[s1, "stage2_reason"] = out[s1].apply(stage2_reason, axis=1)
    out["stage2_pass"] = s1 & (out.stage2_reason == "")
    out["selected"] = False
    rng = np.random.default_rng(SELECT_SEED)
    for group, n in QUOTA.items():
        ids = np.sort(out.loc[out.stage2_pass & (out.group == group), "id"].to_numpy())
        if len(ids) < n:
            print(f"  {group}: only {len(ids)} pass stage 2, need {n}; all taken")
        pick = ids if len(ids) <= n else rng.choice(ids, n, replace=False)
        out.loc[out.id.isin(pick), "selected"] = True
    out.to_csv(SELECTION_CSV, index=False)

    flow = out.groupby("group").agg(recorded=("id", "size"), stage1=("usable", "sum"),
                                    stage2=("stage2_pass", "sum"), selected=("selected", "sum"))
    flow.loc["all"] = flow.sum()
    print("flow per group:")
    print(flow.to_string())
    reasons = out[s1 & ~out.stage2_pass].groupby(["group", "stage2_reason"]).size()
    print("\nstage-2 exclusions per group and reason:")
    print(reasons.to_string() if len(reasons) else "  none")

    sel = out[out.selected]
    cols = ["base_level_uS", "base_scr_pm", "base_scr_med_uS", "base_noise_uS", "base_snr"]
    rows = []
    for c in cols:
        per = {g: float(v.median()) for g, v in sel.groupby("group")[c]}
        p = kruskal(*[v.dropna() for _, v in sel.groupby("group")[c]]).pvalue
        rows.append({"measure": c, **{f"median_{g}": v for g, v in per.items()}, "kruskal_p": p})
    bal = pd.DataFrame(rows)
    bal.to_csv(BALANCE_CSV, index=False)
    print("\nselected: baseline medians per group (Kruskal-Wallis p; small p = groups differ)")
    print(bal.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("saved", SELECTION_CSV.name, BALANCE_CSV.name)
    return out


TUNE_SELECTED_CSV = cfg.ROOT / "data" / "own_tune_scores_selected.csv"


def score_selected():
    """own-score-selected: the tuning scores again, over the tuning sessions whose source
    participant passes stage 2 only.  Does the frozen setting still rank at the top for the
    population we analyse?  Reads own_tune_scores.csv (own-score) and selection.csv (select)."""
    from . import tbme_synth as ts
    from . import tuning
    s = pd.read_csv(tuning.TUNE_CSV)
    src = pd.read_csv(ts.COHORT_CSV)[["session", "source_id"]].astype({"source_id": int})
    sel = pd.read_csv(SELECTION_CSV)[["id", "stage2_pass"]].rename(columns={"id": "source_id"})
    s = s.merge(src, on="session").merge(sel, on="source_id", how="left")
    kept = s[s.stage2_pass.fillna(False).astype(bool)]
    n = kept.drop_duplicates("session").groupby("tau_group").size()
    print(f"tuning sessions whose source passes stage 2: {kept.session.nunique()} of "
          f"{s.session.nunique()} ({', '.join(f'{g} {v}' for g, v in n.items())})")
    cols = ["f1_visible", "recall_visible", "precision", "timing_err_s", "amp_spearman",
            "tau_rise_err_pct", "tau_decay_err_pct", "windows_authors_rule"]
    t = kept.groupby("setting")[cols].median().sort_values("f1_visible", ascending=False)
    t.to_csv(TUNE_SELECTED_CSV)
    print("median over those sessions, best f1_visible first:")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print("saved", TUNE_SELECTED_CSV.name)
    return t
