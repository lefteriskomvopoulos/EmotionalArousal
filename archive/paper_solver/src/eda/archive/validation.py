"""
Stage 1 validated against synthetic sessions with a known answer.

Each session from synthetic.cohort goes through the unchanged stage-1 steps (tonic removal,
kernel fit, FISTA, event merging), and the result is scored against the truth:

    detection   true impulses matched to recovered ones within cfg.MERGE_WINDOW_S
                -> recall, precision, F1, timing error
    amplitude   recovered vs true size of each matched impulse
    kernel      fitted vs true time constants, and whether a fit landed on a bound
    tonic       how far the percentile floor is from the true drift

Impulses below cfg.MIN_EVENT_uS are discarded by design, so recall is also reported over the
true impulses the pipeline is allowed to see.
"""
import numpy as np
import pandas as pd

from .. import config as cfg
from . import deconvolution as dec
from ..tbme_score import match
from .synthetic import cohort, realistic_cohort

VALIDATION_CSV = cfg.ROOT / "data" / "synthetic_validation.csv"


def deconvolve_session(s, lam=None, oracle=False, cost_type=None, debias=None,
                       tonic_pct=None, tonic_win_s=None):
    """Stage 1 exactly as deconvolution.run does it, minus loading and downsampling.

    With oracle=True the true tonic and the true kernel are handed in, so the only thing left
    to go wrong is the sparse solve itself, i.e. the effect of lam alone.  The other keywords
    override the matching config values, so fixes can be compared without editing config.py.
    """
    debias = cfg.DEBIAS if debias is None else debias
    if oracle:
        tonic = s["tonic"]
        phasic = s["eda"] - tonic
        tau_r, tau_d = s["tau_rise"], s["tau_decay"]
    else:
        tonic, phasic = dec.split_tonic_phasic(s["eda"], pct=tonic_pct, win_s=tonic_win_s)
        tau_r, tau_d = cfg.POOLED_KERNEL or dec.fit_kernel(phasic, lam=lam, cost_type=cost_type)
    h = dec.kernel(tau_r, tau_d)
    u = dec.deconvolve(phasic, h, lam=lam)
    fit = np.convolve(u, h)[:len(phasic)]
    ev_t, ev_a = dec.extract_events(u, t0=s["t"][0])
    if debias:
        ev_a, fit = dec.debias(phasic, h, ev_t, t0=s["t"][0])
    return {"tonic": tonic, "phasic": phasic, "tau_rise": tau_r, "tau_decay": tau_d,
            "ev_t": ev_t, "ev_a": ev_a, "r2": 1 - (phasic - fit).var() / phasic.var()}


def score(s, r):
    """Compare one recovered session r with its truth s -> one row of metrics."""
    ti, ei = match(s["ev_t"], r["ev_t"])
    n_true, n_est, n_hit = len(s["ev_t"]), len(r["ev_t"]), len(ti)
    visible = s["ev_a"] >= cfg.MIN_EVENT_uS
    recall = n_hit / max(n_true, 1)
    precision = n_hit / max(n_est, 1)
    a_true, a_est = s["ev_a"][ti], r["ev_a"][ei]
    rise_pin, decay_pin = dec.tau_pinned(r["tau_rise"], r["tau_decay"])

    return {
        "tau_rise_true": s["tau_rise"], "tau_rise_fit": r["tau_rise"],
        "tau_decay_true": s["tau_decay"], "tau_decay_fit": r["tau_decay"],
        "tau_rise_err_pct": 100 * (r["tau_rise"] / s["tau_rise"] - 1),
        "tau_decay_err_pct": 100 * (r["tau_decay"] / s["tau_decay"] - 1),
        "rise_pinned": rise_pin, "decay_pinned": decay_pin,
        "n_true": n_true, "n_est": n_est, "n_hit": n_hit,
        "recall": recall, "precision": precision,
        "f1": 2 * recall * precision / max(recall + precision, 1e-12),
        "recall_visible": np.isin(np.flatnonzero(visible), ti).mean() if visible.any() else np.nan,
        "timing_err_s": np.abs(s["ev_t"][ti] - r["ev_t"][ei]).mean() if n_hit else np.nan,
        "amp_corr": np.corrcoef(a_true, a_est)[0, 1] if n_hit > 2 else np.nan,
        "amp_err_pct_median": 100 * np.median(a_est / a_true - 1) if n_hit else np.nan,
        "hit_amp_median": np.median(a_est) if n_hit else np.nan,
        "false_amp_median": np.median(np.delete(r["ev_a"], ei)) if n_est > n_hit else np.nan,
        "tonic_rmse_uS": np.sqrt(np.mean((r["tonic"] - s["tonic"]) ** 2)),
        "r2": r["r2"],
    }


def run(n_subjects=20, seed=0, **kw):
    """Simulate, deconvolve and score a synthetic cohort; write the table and print a summary."""
    rows = []
    for i, s in enumerate(cohort(n_subjects, seed=seed, **kw)):
        row = {"session": i, **score(s, deconvolve_session(s))}
        rows.append(row)
        print(f"S{i:02d}: tau {row['tau_rise_true']:.2f}/{row['tau_decay_true']:.2f} -> "
              f"{row['tau_rise_fit']:.2f}/{row['tau_decay_fit']:.2f} s | "
              f"recall {row['recall']:.2f} precision {row['precision']:.2f} | "
              f"amp r {row['amp_corr']:.2f} | R2 {row['r2']:.3f}")

    v = pd.DataFrame(rows)
    v.to_csv(VALIDATION_CSV, index=False)

    print(f"\n{n_subjects} sessions -> {VALIDATION_CSV.name}  (median [IQR])")
    for col in ("recall", "recall_visible", "precision", "f1", "timing_err_s", "amp_corr",
                "amp_err_pct_median", "tau_rise_err_pct", "tau_decay_err_pct",
                "tonic_rmse_uS", "r2"):
        q1, q2, q3 = v[col].quantile([0.25, 0.5, 0.75])
        print(f"   {col:20s} {q2:8.3f}  [{q1:.3f}, {q3:.3f}]")
    print(f"   pinned: rise {v['rise_pinned'].sum()}, decay {v['decay_pinned'].sum()} "
          f"of {n_subjects}")
    return v


SWEEP_CSV = cfg.ROOT / "data" / "synthetic_lambda_sweep.csv"


def lambda_sweep(lams=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6), n_subjects=10, seed=0, **kw):
    """Score the same synthetic sessions at several lam values, in two conditions.

    oracle   true tonic and true kernel: isolates what lam does to detection and amplitude
    full     the whole pipeline, kernel fit included, since lam also enters the kernel cost

    Comparing the two at each lam shows how much of the error is lam itself and how much
    comes from the tonic floor and the kernel fit.
    """
    sessions = cohort(n_subjects, seed=seed, **kw)
    rows = []
    for lam in lams:
        for mode in ("oracle", "full"):
            for i, s in enumerate(sessions):
                r = deconvolve_session(s, lam=lam, oracle=(mode == "oracle"))
                rows.append({"lam": lam, "mode": mode, "session": i, **score(s, r)})
            print(f"lam {lam:<5} {mode:6s} done")

    v = pd.DataFrame(rows)
    v.to_csv(SWEEP_CSV, index=False)

    cols = ["recall", "precision", "f1", "amp_err_pct_median", "amp_corr",
            "tau_rise_err_pct", "tau_decay_err_pct", "r2"]
    with pd.option_context("display.width", 160, "display.precision", 2):
        print(f"\nmedians over {n_subjects} sessions -> {SWEEP_CSV.name}")
        print(v.groupby(["mode", "lam"])[cols].median())
    return v


FIXES_CSV = cfg.ROOT / "data" / "synthetic_fixes.csv"

# Each variant adds one change to the one before it, so every step's effect can be read off.
VARIANTS = {
    "A current":           {},
    "B sse kernel":        {"cost_type": "sse"},
    "C B + debias":        {"cost_type": "sse", "debias": True},
    "D C + tonic pct 1":   {"cost_type": "sse", "debias": True, "tonic_pct": 1},
    "E C + tonic win 20s": {"cost_type": "sse", "debias": True, "tonic_win_s": 20},
}


def compare_fixes(lams=(0.2, 0.4), n_subjects=10, seed=0, **kw):
    """Run the full pipeline on the same synthetic sessions under each variant in VARIANTS."""
    sessions = cohort(n_subjects, seed=seed, **kw)
    rows = []
    for lam in lams:
        for name, opts in VARIANTS.items():
            for i, s in enumerate(sessions):
                r = deconvolve_session(s, lam=lam, **{"debias": False, **opts})
                rows.append({"lam": lam, "variant": name, "session": i, **score(s, r)})
            print(f"lam {lam:<4} {name} done")

    v = pd.DataFrame(rows)
    v.to_csv(FIXES_CSV, index=False)

    cols = ["recall_visible", "precision", "amp_err_pct_median", "amp_corr",
            "tau_rise_err_pct", "tau_decay_err_pct", "tonic_rmse_uS"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 2):
        print(f"\nmedians over {n_subjects} sessions -> {FIXES_CSV.name}")
        print(v.groupby(["lam", "variant"])[cols].median())
        print("\nfits pinned to a bound (rise, decay):")
        print(v.groupby(["lam", "variant"])[["rise_pinned", "decay_pinned"]].sum())
    return v


ROBUST_CSV = cfg.ROOT / "data" / "synthetic_robustness.csv"


def robustness(amp_medians=(0.03, 0.06, 0.12), lams=(0.2, 0.4, 0.8), n_subjects=10, seed=0):
    """Current pipeline with and without debiasing, across burst sizes and lam values.

    The penalised kernel fit came out unbiased near lam = 0.4 for 0.06 uS bursts.  Since lam
    trades against amplitude, that sweet spot may move with burst size (equivalently with
    signal-to-noise, the noise being fixed), which is what this checks.  Debiasing only changes
    the amplitudes, so both versions share one kernel fit and one set of event times.
    """
    rows = []
    for amp in amp_medians:
        sessions = cohort(n_subjects, seed=seed, amp_median_uS=amp)
        for lam in lams:
            for i, s in enumerate(sessions):
                r = deconvolve_session(s, lam=lam, debias=False)
                h = dec.kernel(r["tau_rise"], r["tau_decay"])
                rd = {**r, "ev_a": dec.debias(r["phasic"], h, r["ev_t"], t0=s["t"][0])[0]}
                for name, res in (("raw", r), ("debiased", rd)):
                    rows.append({"amp_median_uS": amp, "lam": lam, "amps": name,
                                 "session": i, **score(s, res)})
            print(f"amp {amp:<5} lam {lam:<4} done")

    v = pd.DataFrame(rows)
    v.to_csv(ROBUST_CSV, index=False)

    cols = ["recall_visible", "precision", "amp_err_pct_median", "amp_corr",
            "tau_rise_err_pct", "tau_decay_err_pct"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 2):
        print(f"\nmedians over {n_subjects} sessions -> {ROBUST_CSV.name}")
        print(v.groupby(["amp_median_uS", "lam", "amps"])[cols].median())
        print("\nfits pinned to a bound (rise, decay), out of", n_subjects)
        print(v[v["amps"] == "raw"].groupby(["amp_median_uS", "lam"])
              [["rise_pinned", "decay_pinned"]].sum())
    print(f"\nmissing values in table: {int(v.isna().sum().sum())}")
    return v


GCV_CSV = cfg.ROOT / "data" / "synthetic_gcv.csv"


def gcv_test(amp_medians=(0.03, 0.06, 0.12), n_subjects=10, seed=0):
    """Per-session lam by GCV, alternating with the kernel fit, then debiased amplitudes.

    Same sessions as robustness(), so the GCV row for each burst size can be read against the
    fixed-lam rows there.  The question is whether GCV lands near the lam that gave an unbiased
    kernel (about 0.2, 0.3 and 0.8 for 0.03, 0.06 and 0.12 uS bursts).
    """
    rows = []
    for amp in amp_medians:
        for i, s in enumerate(cohort(n_subjects, seed=seed, amp_median_uS=amp)):
            tonic, phasic = dec.split_tonic_phasic(s["eda"])
            tau_r, tau_d, lam, rounds = dec.fit_kernel_and_lambda(phasic)
            h = dec.kernel(tau_r, tau_d)
            u = dec.deconvolve(phasic, h, lam=lam)
            ev_t, _ = dec.extract_events(u, t0=s["t"][0])
            ev_a, fit = dec.debias(phasic, h, ev_t, t0=s["t"][0])
            r = {"tonic": tonic, "tau_rise": tau_r, "tau_decay": tau_d, "ev_t": ev_t,
                 "ev_a": ev_a, "r2": 1 - (phasic - fit).var() / phasic.var()}
            rows.append({"amp_median_uS": amp, "session": i, "lam": lam, "rounds": rounds,
                         **score(s, r)})
            print(f"amp {amp:<5} S{i:02d}: lam {lam:<5} ({rounds} rounds) | "
                  f"tau {s['tau_rise']:.2f}/{s['tau_decay']:.2f} -> {tau_r:.2f}/{tau_d:.2f} s")

    v = pd.DataFrame(rows)
    v.to_csv(GCV_CSV, index=False)

    cols = ["lam", "recall_visible", "precision", "amp_err_pct_median", "amp_corr",
            "tau_rise_err_pct", "tau_decay_err_pct"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 2):
        print(f"\nmedians over {n_subjects} sessions -> {GCV_CSV.name}")
        print(v.groupby("amp_median_uS")[cols].median())
        print("\nfits pinned to a bound (rise, decay):")
        print(v.groupby("amp_median_uS")[["rise_pinned", "decay_pinned"]].sum())
    print(f"\nmissing values in table: {int(v.isna().sum().sum())}")
    return v


SCALE_CSV = cfg.ROOT / "data" / "synthetic_scale.csv"


def scale_test(cs=(2, 4, 8, 16), n_simple=10, n_realistic=20, seed=0):
    """lam = c x burst_scale(phasic), with debiased amplitudes, on two synthetic sets.

    simple      the cohort() sessions at 0.03 / 0.06 / 0.12 uS, as in robustness()
    realistic   realistic_cohort(): every property varies, correlated noise, artefacts

    Besides the usual medians, the invariance check: the Spearman correlation between true
    burst size and the tau_decay error.  Under a fixed lam it is strongly negative (small
    responders get slow kernels); a scale rule that works should bring it close to zero.
    """
    sets = [("simple", s) for amp in (0.03, 0.06, 0.12)
            for s in cohort(n_simple, seed=seed, amp_median_uS=amp)]
    sets += [("realistic", s) for s in realistic_cohort(n_realistic)]

    rows = []
    for c in cs:
        for i, (name, s) in enumerate(sets):
            tonic, phasic = dec.split_tonic_phasic(s["eda"])
            scale = dec.burst_scale(phasic)
            lam = c * scale
            tau_r, tau_d = dec.fit_kernel(phasic, lam=lam)
            h = dec.kernel(tau_r, tau_d)
            ev_t, _ = dec.extract_events(dec.deconvolve(phasic, h, lam=lam), t0=s["t"][0])
            ev_a, fit = dec.debias(phasic, h, ev_t, t0=s["t"][0])
            r = {"tonic": tonic, "tau_rise": tau_r, "tau_decay": tau_d, "ev_t": ev_t,
                 "ev_a": ev_a, "r2": 1 - (phasic - fit).var() / phasic.var()}
            rows.append({"c": c, "set": name, "session": i,
                         "amp_median_true": s["amp_median_uS"], "scale_uS": scale, "lam": lam,
                         **score(s, r)})
        print(f"c {c:<3} done")

    v = pd.DataFrame(rows)
    v.to_csv(SCALE_CSV, index=False)

    cols = ["lam", "recall_visible", "precision", "amp_err_pct_median", "amp_corr",
            "tau_rise_err_pct", "tau_decay_err_pct"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 2):
        print(f"\nmedians -> {SCALE_CSV.name}")
        print(v.groupby(["set", "c"])[cols].median())
        print("\nabsolute tau_decay error, median %:")
        print(v.assign(e=v["tau_decay_err_pct"].abs()).groupby(["set", "c"])["e"].median())
        print("\ninvariance: Spearman r(true burst size, tau_decay error), want ~0:")
        print(v.groupby(["set", "c"]).apply(
            lambda g: g["amp_median_true"].corr(g["tau_decay_err_pct"], method="spearman")))
        print("\nscale measure vs true burst size, Spearman r:",
              round(v[v["c"] == cs[0]]["amp_median_true"]
                    .corr(v[v["c"] == cs[0]]["scale_uS"], method="spearman"), 2))
        print("\nfits pinned to a bound (rise, decay):")
        print(v.groupby(["set", "c"])[["rise_pinned", "decay_pinned"]].sum())
    print(f"\nmissing values in table: {int(v.isna().sum().sum())}")
    return v


CVX_CSV = cfg.ROOT / "data" / "synthetic_cvxeda.csv"


def cvx_sets(n_simple=10, n_realistic=20, seed=0):
    """The same sessions scale_test() used: simple at three burst sizes, then realistic."""
    sets = [("simple", s) for amp in (0.03, 0.06, 0.12)
            for s in cohort(n_simple, seed=seed, amp_median_uS=amp)]
    return sets + [("realistic", s) for s in realistic_cohort(n_realistic)]


def cvx_test(smoke=False):
    """cvxEDA as published on the synthetic sessions, scored like our own pipeline.

    smoke=True runs one session only and reports its time, to check the installation and to
    estimate how long the full set and the real cohort will take.
    """
    import time
    from . import cvx
    sets = cvx_sets()[:1] if smoke else cvx_sets()
    rows = []
    for i, (name, s) in enumerate(sets):
        t0 = time.time()
        r = cvx.decompose(s["eda"], t0=s["t"][0])
        sec = time.time() - t0
        rows.append({"set": name, "session": i, "amp_median_true": s["amp_median_uS"],
                     "seconds": sec, **score(s, r)})
        row = rows[-1]
        print(f"{name:9s} S{i:02d} amp {s['amp_median_uS']:.3f}: recall {row['recall']:.2f} "
              f"precision {row['precision']:.2f} | amp err {row['amp_err_pct_median']:+.0f}% | "
              f"R2 {row['r2']:.3f} | {sec:.0f} s")

    v = pd.DataFrame(rows)
    if smoke:
        return v
    v.to_csv(CVX_CSV, index=False)
    cols = ["recall_visible", "precision", "f1", "timing_err_s", "amp_err_pct_median",
            "amp_corr", "tonic_rmse_uS", "r2", "seconds"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 3):
        print(f"\nmedians -> {CVX_CSV.name}")
        v["burst"] = np.where(v["set"] == "simple", v["amp_median_true"].round(2).astype(str),
                              "mixed")
        print(v.groupby(["set", "burst"])[cols].median())
        print("\ninvariance: Spearman r(true burst size, recall), want ~0:")
        print(v.groupby("set")[["amp_median_true", "recall"]]
              .apply(lambda g: g["amp_median_true"].corr(g["recall"], method="spearman")))
    print(f"\nmissing values in table: {int(v.isna().sum().sum())}")
    return v


CVX_PARAMS_CSV = cfg.ROOT / "data" / "synthetic_cvxeda_params.csv"


def cvx_params_test():
    """cvxEDA under each setting in cvx.CONFIGS, plus the true tau per session as a best case."""
    from . import cvx
    rows = []
    for i, (name, s) in enumerate(cvx_sets()):
        configs = {**cvx.CONFIGS,
                   "true tau/10s": {"tau0": s["tau_decay"], "tau1": s["tau_rise"]},
                   "true tau/30s": {"tau0": s["tau_decay"], "tau1": s["tau_rise"],
                                    "delta_knot": 30.}}
        for conf, kw in configs.items():
            r = cvx.decompose(s["eda"], t0=s["t"][0], **kw)
            rows.append({"set": name, "session": i, "config": conf,
                         "amp_median_true": s["amp_median_uS"], **score(s, r)})
        print(f"{name:9s} S{i:02d} done")

    v = pd.DataFrame(rows)
    v.to_csv(CVX_PARAMS_CSV, index=False)
    cols = ["recall_visible", "precision", "f1", "amp_err_pct_median", "amp_corr",
            "tonic_rmse_uS", "r2"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 2):
        print(f"\nmedians -> {CVX_PARAMS_CSV.name}")
        print(v.groupby(["set", "config"], sort=False)[cols].median())
        print("\ninvariance: Spearman r(true burst size, precision), want ~0:")
        print(v.groupby(["set", "config"], sort=False)[["amp_median_true", "precision"]]
              .apply(lambda g: g["amp_median_true"].corr(g["precision"], method="spearman"))
              .round(2))
    print(f"\nmissing values in table: {int(v.isna().sum().sum())}")
    return v


CVX_ALPHA_CSV = cfg.ROOT / "data" / "synthetic_cvxeda_alpha.csv"
ALPHAS = tuple(8e-4 * 2 ** k for k in range(8))     # published default, then x2 up to ~0.1
CUTS = (0.02, 0.03, 0.05)                           # minimum event amplitude, uS


def cvx_alpha_test():
    """cvxEDA's sparsity weight alpha, at the default tau and at the true tau, on all sessions.

    alpha is chosen on the realistic set only (the one place the truth is known and the
    conditions are hard); if the default-tau and true-tau runs agree on it, the choice does not
    hinge on getting tau right.  Each solution is also scored at three amplitude cut-offs, to
    see whether false impulses are small enough that the cut-off alone would remove them.
    """
    from . import cvx
    rows = []
    for i, (name, s) in enumerate(cvx_sets()):
        for tau_name, kw in (("default", {}),
                             ("true", {"tau0": s["tau_decay"], "tau1": s["tau_rise"]})):
            for alpha in ALPHAS:
                r = cvx.decompose(s["eda"], t0=s["t"][0], alpha=alpha, **kw)
                for cut in CUTS:
                    keep = r["ev_a"] >= cut
                    rc = {**r, "ev_t": r["ev_t"][keep], "ev_a": r["ev_a"][keep]}
                    rows.append({"set": name, "session": i, "tau": tau_name, "alpha": alpha,
                                 "cut": cut, "amp_median_true": s["amp_median_uS"],
                                 **score(s, rc)})
        print(f"{name:9s} S{i:02d} done")

    v = pd.DataFrame(rows)
    v.to_csv(CVX_ALPHA_CSV, index=False)
    cols = ["recall_visible", "precision", "f1", "amp_err_pct_median", "hit_amp_median",
            "false_amp_median"]
    base = v[v["cut"] == CUTS[0]]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 3):
        print(f"\nmedians at cut {CUTS[0]} uS -> {CVX_ALPHA_CSV.name}")
        print(base.groupby(["set", "tau", "alpha"])[cols].median())
        print("\ninvariance: Spearman r(true burst size, precision), realistic set:")
        print(base[base["set"] == "realistic"].groupby(["tau", "alpha"])
              [["amp_median_true", "precision"]]
              .apply(lambda g: g["amp_median_true"].corr(g["precision"], method="spearman"))
              .round(2).unstack("tau"))
        print("\nrealistic set, median F1 by alpha (rows) and amplitude cut-off (columns):")
        print(v[v["set"] == "realistic"].groupby(["tau", "alpha", "cut"])["f1"].median()
              .unstack("cut"))
    print(f"\nmissing values in table: {int(v[['recall', 'precision', 'f1']].isna().sum().sum())}")
    return v


CVX_TAU_CSV = cfg.ROOT / "data" / "synthetic_cvxeda_tau.csv"


def cvx_tau_test(costs=("obj", "sse"), alphas=(6.4e-3, 1.28e-2, 2.56e-2)):
    """Per-session tau by the outer loop (cvx.fit_tau), then scored like everything else.

    tau is fitted once per session and cost at the provisional alpha, then cvxEDA is re-run at
    three alphas around it with the fitted tau: the alpha check with estimated tau.  The
    true-tau rows at the same alphas are the ceiling to compare against.
    """
    from . import cvx
    rows = []
    for i, (name, s) in enumerate(cvx_sets()):
        fits = {c: cvx.fit_tau(s["eda"], cost=c) for c in costs}
        fits["true"] = (s["tau_rise"], s["tau_decay"])
        for how, (t1, t0) in fits.items():
            pin = cvx.tau_pinned(t1, t0) if how != "true" else (False, False)
            for alpha in alphas:
                r = cvx.decompose(s["eda"], t0=s["t"][0], tau0=t0, tau1=t1, alpha=alpha,
                                  cut=cvx.CUT_uS)
                row = score(s, r)
                row.update(rise_pinned=pin[0], decay_pinned=pin[1])
                rows.append({"set": name, "session": i, "tau_from": how, "alpha": alpha,
                             "amp_median_true": s["amp_median_uS"], **row})
        print(f"{name:9s} S{i:02d}: true {s['tau_rise']:.2f}/{s['tau_decay']:.2f} | " +
              " | ".join(f"{c} {fits[c][0]:.2f}/{fits[c][1]:.2f}" for c in costs))

    v = pd.DataFrame(rows)
    v.to_csv(CVX_TAU_CSV, index=False)
    v["abs_tau_decay_err_pct"] = v["tau_decay_err_pct"].abs()
    cols = ["recall_visible", "precision", "f1", "amp_err_pct_median", "tau_rise_err_pct",
            "tau_decay_err_pct", "abs_tau_decay_err_pct"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.precision", 3):
        print(f"\nmedians -> {CVX_TAU_CSV.name}")
        print(v.groupby(["set", "tau_from", "alpha"])[cols].median())
        print("\ninvariance, Spearman r with true burst size (want ~0):")
        g = v.groupby(["set", "tau_from", "alpha"])
        print(pd.DataFrame({
            "precision": g.apply(lambda d: d["amp_median_true"].corr(d["precision"],
                                                                     method="spearman")),
            "tau_decay_err": g.apply(lambda d: d["amp_median_true"].corr(
                d["tau_decay_err_pct"], method="spearman")),
        }).round(2))
        print("\nfits pinned to a search bound (rise, decay), per set and cost:")
        one = v[(v["alpha"] == alphas[0]) & (v["tau_from"] != "true")]
        print(one.groupby(["set", "tau_from"])[["rise_pinned", "decay_pinned"]].sum())
    print(f"\nmissing values: {int(v[['recall', 'precision', 'f1']].isna().sum().sum())}")
    return v
