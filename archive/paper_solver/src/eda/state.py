"""
Stage 2: estimating the hidden arousal state from the recovered impulses.

Marked point process filter, following Wickramasuriya & Faghih, IEEE Access 8:68499, 2020.

    x_k = x_{k-1} + noise,                       noise ~ N(0, sv)        the hidden arousal state
    P(n_k = 1 | x_k) = sigmoid(b0 + x_k)                                 did a burst occur?
    r_k | n_k = 1    ~ N(g0 + g1 * x_k, sr)                              how big was it (log uS)?

So arousal raises both how often bursts happen and how large they are.  The state is estimated by
a forward filter and a backward smoother; the five parameters are learned by
expectation-maximization.

Supporting references
    Eden et al., Neural Computation 16:971, 2004   - Newton update for point-process filters
    Smith & Brown, Neural Computation 15:965, 2003 - EM for state-space models with binary data
"""
import numpy as np

from . import config as cfg


def to_bins(ev_t, ev_a, duration, t0=0.0):
    """Impulse list -> per-bin observations (n, r).

    n is 1 where a burst occurred, r is its log amplitude (NaN in empty bins).  Bin 0 starts at
    t0, the first timestamp of the recording, not at absolute zero.

    Known limitation: the model allows at most one impulse per bin, so when two land in the same
    bin only the larger is kept.  At cfg.BIN_S = 5 s that happens in roughly a quarter of occupied
    bins, which is on the list to fix.
    """
    K = int(duration / cfg.BIN_S)
    n = np.zeros(K, dtype=int)
    r = np.full(K, np.nan)
    for t, a in zip(ev_t, ev_a):
        k = min(int((t - t0) / cfg.BIN_S), K - 1)
        if n[k] == 0 or a > np.exp(r[k]):
            n[k], r[k] = 1, np.log(a)
    return n, r


def forward_filter(n, r, b0, g0, g1, sr, sv):
    """One forward pass over the bins: predict, then correct.

    Predicting carries the previous estimate over and widens its uncertainty by sv.  Correcting
    combines that prior with whatever the bin showed.  Because the burst/no-burst observation is
    Bernoulli rather than Gaussian, the correction has no closed form, so we locate the mode of
    the log-posterior with a few Newton steps.  The curvature at the mode gives the variance.

    Returns (x_filtered, v_filtered, x_predicted, v_predicted).
    """
    K = len(n)
    xf, vf, xp, vp = (np.zeros(K) for _ in range(4))
    x_prev, v_prev = 0.0, 1.0

    for k in range(K):
        xp[k], vp[k] = x_prev, v_prev + sv                # predict
        x = xp[k]
        for _ in range(10):                               # correct
            p = 1 / (1 + np.exp(-(b0 + x)))
            grad = -(x - xp[k]) / vp[k] + (n[k] - p)
            hess = -1 / vp[k] - p * (1 - p)
            if n[k]:
                grad += g1 * (r[k] - g0 - g1 * x) / sr
                hess += -g1 ** 2 / sr
            x -= grad / hess
        p = 1 / (1 + np.exp(-(b0 + x)))
        hess = -1 / vp[k] - p * (1 - p) - (g1 ** 2 / sr if n[k] else 0)
        xf[k], vf[k] = x, -1 / hess
        x_prev, v_prev = xf[k], vf[k]

    return xf, vf, xp, vp


def rts_smoother(xf, vf, xp, vp):
    """Backward pass (Rauch-Tung-Striebel): re-estimate each bin using the whole recording.

    Also returns the lag-one covariance W[k] = cov(x_k, x_{k-1} | all data).  The EM update for
    the drift rate needs it; leaving it out underestimates the drift and collapses the state into
    a straight line.
    """
    K = len(xf)
    xs, vs = xf.copy(), vf.copy()
    A = np.zeros(K)
    for k in range(K - 2, -1, -1):
        A[k] = vf[k] / vp[k + 1]
        xs[k] = xf[k] + A[k] * (xs[k + 1] - xp[k + 1])
        vs[k] = vf[k] + A[k] ** 2 * (vs[k + 1] - vp[k + 1])
    W = np.zeros(K)
    W[1:] = A[:-1] * vs[1:]
    return xs, vs, W


def fit_state(n, r, iters=None):
    """Expectation-maximization over the state and the five model parameters.

    E step: estimate the state with the current parameters (filter, then smoother).
    M step: re-fit the parameters to that state.  g0 and g1 come from a least-squares line through
    the (estimated arousal, log amplitude) pairs; sv comes from how much the state moved.

    Two safeguards, both documented in config:
      * g1 is clipped to [G1_MIN, G1_MAX].  The lower bound is physiology, since arousal cannot
        make bursts smaller; unconstrained, the regression once fitted -0.58 and inverted the
        whole estimate.  The upper bound is for stability.
      * the drift-rate update adds back the variance that smoothing removes.  Measuring the wiggle
        of an already-smoothed curve underestimates it and drives sv toward zero.

    b0 is held fixed at the overall burst log-odds: only b0 + x enters the model, so one of them
    must be pinned or the state's level is unidentifiable.

    Returns (state, standard_deviation, parameters).
    """
    iters = cfg.EM_ITERS if iters is None else iters

    b0 = np.log(max(n.mean(), 1e-3) / max(1 - n.mean(), 1e-3))
    g0, g1 = np.nanmean(r), 0.5
    sr = cfg.SR_FIXED
    sv = 0.05

    for _ in range(iters):
        # --- E step ---------------------------------------------------------------------
        xf, vf, xp, vp = forward_filter(n, r, b0, g0, g1, sr, sv)
        xs, vs, W = rts_smoother(xf, vf, xp, vp)

        # --- M step ---------------------------------------------------------------------
        d = xs[1:] - xs[:-1]
        sv = max(np.mean(d ** 2 + vs[1:] + vs[:-1] - 2 * W[1:]), 1e-4)

        m = ~np.isnan(r)                                  # bins that contained a burst
        if m.sum() > 2:
            A = np.c_[np.ones(m.sum()), xs[m]]
            _, g1_hat = np.linalg.lstsq(A, r[m], rcond=None)[0]
            g1 = float(np.clip(g1_hat, cfg.G1_MIN, cfg.G1_MAX))
            g0 = r[m].mean() - g1 * xs[m].mean()          # refit the intercept for that slope
            sr = cfg.SR_FIXED                             # fixed, not learned

    return xs, np.sqrt(np.maximum(vs, 0)), dict(b0=b0, g0=g0, g1=g1, sr=sr, sv=sv)


def run(dec):
    """Stage 2 for one participant, given the output of deconvolution.run().

    Returns the estimated state, its standard deviation, the bin centres on the recording's own
    clock, and the fitted parameters.
    """
    t = dec["t"]
    n, r = to_bins(dec["ev_t"], dec["ev_a"], t[-1] - t[0], t0=t[0])
    xs, sd, par = fit_state(n, r)
    tb = t[0] + (np.arange(len(xs)) + 0.5) * cfg.BIN_S
    return {"n": n, "r": r, "state": xs, "state_sd": sd, "t_bins": tb, "params": par}
