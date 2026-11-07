"""
Stage 1 by cvxEDA (Greco et al., IEEE TBME 63:797, 2016), run exactly as published.

The authors' code is used unmodified: external/cvxEDA, github.com/lciti/cvxEDA at commit
7f518fe (13 Feb 2026), GPL-3.0, with every parameter at its published default
(tau0 = 2 s, tau1 = 0.7 s, delta_knot = 10 s, alpha = 8e-4, gamma = 1e-2, linear baseline).
Unlike our pipeline, cvxEDA fits the tonic level jointly with the phasic part (splines) and
keeps the Bateman time constants fixed for everybody.

Only the interface is ours, and each step is stated so the report can say so:
    input    the same 10 Hz signal as deconvolution.to_deconv_rate, z-scored as the authors
             recommend ("we recommend normalizing it: y = zscore(y)")
    units    outputs are mapped back to uS by undoing the z-score; the driver p is scaled by the
             peak of cvxEDA's own Bateman response, so an event amplitude means the same as in
             our pipeline (peak height of its response, uS)
    events   p comes from an interior-point solver, so it is small but not exactly zero between
             bursts.  Events are its local maxima at least cfg.MERGE_WINDOW_S apart, with the
             driver summed over that window as amplitude, keeping those >= cfg.MIN_EVENT_uS.
             This mirrors deconvolution.extract_events.
"""
import numpy as np

from .. import config as cfg

TAU0, TAU1 = 2.0, 0.7   # cvxEDA's published defaults (slow, fast), stated here for reporting


def bateman_peak(delta, tau0=TAU0, tau1=TAU1, dur_s=60):
    """Peak of the phasic response cvxEDA produces for a unit driver sample.

    cvxEDA writes r = M q and p = A q, i.e. r = (M/A) p, with A and M built from the
    coefficients below (the same formula as in cvxEDA.py).  Filtering a unit impulse through
    M/A gives the response whose peak converts the driver into peak-height units.
    """
    from scipy.signal import lfilter
    a1, a0 = 1. / min(tau1, tau0), 1. / max(tau1, tau0)
    ar = np.array([(a1 * delta + 2.) * (a0 * delta + 2.), 2. * a1 * a0 * delta ** 2 - 8.,
                   (a1 * delta - 2.) * (a0 * delta - 2.)]) / ((a1 - a0) * delta ** 2)
    imp = np.zeros(int(dur_s / delta))
    imp[0] = 1.0
    return lfilter([1., 2., 1.], ar, imp).max()


def driver_events(u, t0=0.0, cut=None):
    """Local maxima of the driver -> (times, amplitudes); see the module docstring."""
    cut = cfg.MIN_EVENT_uS if cut is None else cut
    from scipy.signal import find_peaks
    span = int(cfg.MERGE_WINDOW_S * cfg.DFS)
    peaks, _ = find_peaks(u, distance=span)
    amps = np.array([u[max(k - span // 2, 0):k + span // 2 + 1].sum() for k in peaks])
    keep = amps >= cut
    return t0 + peaks[keep] / cfg.DFS, amps[keep]


def decompose(eda, t0=0.0, tau0=TAU0, tau1=TAU1, delta_knot=10., alpha=8e-4, cut=None):
    """cvxEDA on one 10 Hz signal (uS) -> the same keys as deconvolution.run produces.

    tau0, tau1, delta_knot and alpha are cvxEDA's own arguments; the defaults are the published
    ones.
    Changing them changes a documented input, not the method.
    """
    import cvxeda
    mu, sd = eda.mean(), eda.std()
    delta = 1.0 / cfg.DFS
    r, p, t, l, d, e, obj = cvxeda.cvxEDA((eda - mu) / sd, delta, tau0=tau0, tau1=tau1,
                                          delta_knot=delta_knot, alpha=alpha,
                                          options={"show_progress": False})   # quiet only

    tonic = t * sd + mu
    phasic = eda - tonic                    # what cvxEDA leaves for the phasic part, + noise
    u = p * sd * bateman_peak(delta, tau0, tau1)   # driver in peak-height uS per sample
    ev_t, ev_a = driver_events(u, t0=t0, cut=cut)
    fit = r * sd
    return {"tonic": tonic, "phasic": phasic, "u": u, "fit": fit,
            "tau_rise": tau1, "tau_decay": tau0, "ev_t": ev_t, "ev_a": ev_a,
            "r2": 1 - (phasic - fit).var() / phasic.var(),
            "obj": float(np.ravel(obj)[0]), "sse_z": float(e @ e)}


def run(pid, load_fn):
    """Stage 1 by cvxEDA for one participant, same input as deconvolution.run."""
    from .deconvolution import to_deconv_rate
    d = to_deconv_rate(load_fn(pid))
    return {**d, **decompose(d["eda"], t0=d["t"][0])}


# Settings compared on synthetic and real data.  Only cvxEDA's own arguments change.  The 2 s
# default decay comes from the EDA literature (Ledalab, Alexander et al. 2005); 6.5 s is what our
# own kernel fit gave, which the synthetic tests showed is inflated at lam = 1.6, so 4 s is
# included in between.  delta_knot 30 s matches the window of our own tonic estimate.
CONFIGS = {
    "default 2s/10s":  {},
    "tau 4s/10s":      {"tau0": 4.0},
    "tau 6.5s/10s":    {"tau0": 6.5},
    "tau 4s/30s":      {"tau0": 4.0, "delta_knot": 30.},
    "tau 6.5s/30s":    {"tau0": 6.5, "delta_knot": 30.},
}


# Provisional, from validation.cvx_alpha_test: with the true tau, alpha 1.28e-2 and a 0.03 uS
# cut-off gave the best realistic-set F1 (0.74) with precision nearly independent of burst size.
ALPHA = 1.28e-2
CUT_uS = 0.03

# Search ranges for the outer tau loop.  Our own rise range; the decay range opens down to 1.5 s
# so that cvxEDA's published 2 s stays reachable.
TAU1_RANGE = cfg.TAU_RISE_RANGE
TAU0_RANGE = (1.5, cfg.TAU_DECAY_RANGE[1])


def fit_tau(eda, alpha=ALPHA, cost="obj", delta_knot=10.):
    """Outer loop: per-participant (tau1, tau0), with cvxEDA solving everything else inside.

    The same scheme as Faghih et al. (2014) and Amin & Faghih (2021): kernel parameters outside,
    a convex solve inside.  cost "obj" scores a candidate by cvxEDA's own objective (eq. 15 of
    the paper, sparsity term included); "sse" by the residual alone.  Our own kernel fit showed
    that these two pull tau in opposite directions, so both are tested.
    Coarse grid first, then Nelder-Mead, as in deconvolution.fit_kernel.  -> (tau1, tau0)
    """
    from scipy.optimize import minimize

    def score(log_taus):
        t1, t0 = np.exp(log_taus)
        if not (TAU1_RANGE[0] <= t1 <= TAU1_RANGE[1] and TAU0_RANGE[0] <= t0 <= TAU0_RANGE[1]
                and t1 < t0):
            return np.inf
        r = decompose(eda, tau0=t0, tau1=t1, alpha=alpha, delta_knot=delta_knot)
        return r["obj"] if cost == "obj" else r["sse_z"]

    grid = [(score(np.log([t1, t0])), t1, t0)
            for t1 in (0.6, 1.0, 1.5) for t0 in (2.0, 4.0, 7.0, 12.0)]
    _, t1, t0 = min(grid)
    best = minimize(score, np.log([t1, t0]), method="Nelder-Mead",
                    options={"xatol": 0.02, "fatol": 1e-4, "maxiter": 30})
    return tuple(np.exp(best.x))


def tau_pinned(tau1, tau0):
    """Whether a fitted time constant ended on its search bound."""
    return (min(abs(tau1 - b) for b in TAU1_RANGE) < 0.02,
            min(abs(tau0 - b) for b in TAU0_RANGE) < 0.05)
