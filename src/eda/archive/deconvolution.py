"""
Stage 1: recovering the sudomotor impulses from skin conductance.

The model is

    EDA(t) = h(t) * u(t) + tonic(t) + noise

where u is a sparse non-negative train of neural impulses, h is the sweat-gland response, and
tonic is a slow drift.  We remove the tonic part, fit h, and solve for u.

References
    Greco et al., cvxEDA, IEEE TBME 63:797, 2016        - the objective (L1 + non-negativity)
    Beck & Teboulle, SIAM J Imaging Sci 2:183, 2009     - FISTA, the solver
    Faghih et al., PLOS ONE 9:e85204, 2014              - alternating impulse / parameter fitting
"""
import numpy as np

from .. import config as cfg


def to_deconv_rate(d):
    """Low-pass filter, then downsample to cfg.DFS Hz.

    Unlike the plotting path, which simply keeps every n-th sample, this applies an anti-aliasing
    filter first.  Without it, fast content folds back into the signal as slow wiggles, and the
    deconvolution would explain those wiggles with impulses.
    """
    from scipy.signal import decimate
    q = int(round(cfg.FS / cfg.DFS))
    eda = decimate(np.nan_to_num(d["eda"], nan=np.nanmean(d["eda"])),
                   q, ftype="fir", zero_phase=True)
    return {"t": d["t"][::q][:len(eda)], "eda": eda,
            "affect": d["affect"][::q][:len(eda)], "marker": d["marker"][::q][:len(eda)]}


def split_tonic_phasic(eda, pct=None, win_s=None, smooth_s=None):
    """Separate the slow drift (tonic) from the bursts (phasic).

    The tonic level is estimated as a rolling low percentile of the signal, then smoothed.
    Bursts only push conductance upward, so the floor of the signal follows the drift underneath
    them.  A high-pass filter would also remove the slow tail of every real burst and push the
    signal below zero, which breaks the non-negativity assumption downstream.
    """
    from scipy.ndimage import percentile_filter, uniform_filter1d
    pct = cfg.TONIC_PCT if pct is None else pct
    win_s = cfg.TONIC_WIN_S if win_s is None else win_s
    smooth_s = cfg.TONIC_SMOOTH_S if smooth_s is None else smooth_s

    tonic = percentile_filter(eda, percentile=pct, size=int(win_s * cfg.DFS))
    tonic = uniform_filter1d(tonic, size=int(smooth_s * cfg.DFS))
    return tonic, eda - tonic


def kernel(tau_rise, tau_decay, dur_s=None):
    """The sweat-gland response to one impulse: fast rise, slow decay, peak normalised to 1.

    h(t) = exp(-t/tau_decay) - exp(-t/tau_rise).  This biexponential ("Bateman") form is the
    standard SCR shape in the EDA literature, used by Alexander 2005, Benedek & Kaernbach 2010
    and cvxEDA.
    """
    dur_s = cfg.KERNEL_S if dur_s is None else dur_s
    t = np.arange(int(dur_s * cfg.DFS)) / cfg.DFS
    h = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    return h / h.max()


def deconvolve(phasic, h, lam=None, iters=400):
    """Sparse non-negative deconvolution by FISTA.

    Minimises  ||phasic - h*u||^2 + lam*sum(u)  subject to u >= 0.

    The constraint holds because nerves can only fire, and the L1 term encodes the fact that
    bursts are discrete events separated by silence.  Each iteration is a gradient step toward
    fitting the data, a shrink by lam, a clip at zero, and a Nesterov momentum step.  The
    convolution operator is never built as a matrix: applying it is np.convolve and its transpose
    is np.correlate.
    """
    lam = cfg.LAM if lam is None else lam
    n, L = len(phasic), len(h)
    forward = lambda u: np.convolve(u, h)[:n]
    adjoint = lambda r: np.correlate(np.r_[r, np.zeros(L - 1)], h, "valid")[:n]
    step = 1.0 / np.abs(h).sum() ** 2        # largest provably safe gradient step

    u = np.zeros(n)
    z = u.copy()
    mom = 1.0
    for _ in range(iters):
        grad = adjoint(forward(z) - phasic)
        u_new = np.maximum(z - step * (grad + lam), 0)
        mom_new = (1 + np.sqrt(1 + 4 * mom ** 2)) / 2
        z = u_new + (mom - 1) / mom_new * (u_new - u)
        u, mom = u_new, mom_new
    return u


def fit_kernel(phasic, seg_s=300, lam=None, cost_type=None):
    """Identify (tau_rise, tau_decay) for one participant.

    For each candidate pair we deconvolve a short segment and score it by how well the recovered
    impulses reproduce the signal, plus the sparsity cost.  A coarse grid first, because the cost
    surface has local minima, then Nelder-Mead refinement.  Candidates outside the physiological
    bounds are rejected: without them the optimiser shrinks the kernel to a narrow spike, which
    fits anything but is not a sweat-gland response.

    cost_type "penalised" scores a candidate by reconstruction error plus lam*sum(u); "sse" by
    reconstruction error alone.  The penalty term rewards slow kernels, which explain the same
    signal with smaller impulses, so on synthetic data the fitted tau_decay followed lam
    (see validation.lambda_sweep).
    """
    from scipy.optimize import minimize
    lam = cfg.LAM if lam is None else lam
    cost_type = cfg.KERNEL_COST if cost_type is None else cost_type
    seg = phasic[:int(seg_s * cfg.DFS)]

    def cost(log_taus):
        tr, td = np.exp(log_taus)
        if not (cfg.TAU_RISE_RANGE[0] <= tr <= cfg.TAU_RISE_RANGE[1]
                and cfg.TAU_DECAY_RANGE[0] <= td <= cfg.TAU_DECAY_RANGE[1] and tr < td):
            return 1e9
        h = kernel(tr, td)
        u = deconvolve(seg, h, lam=lam, iters=150)
        sse = ((seg - np.convolve(u, h)[:len(seg)]) ** 2).sum()
        return sse + lam * u.sum() if cost_type == "penalised" else sse

    grid = [(cost(np.log([tr, td])), tr, td)
            for tr in (0.6, 0.9, 1.3, 1.8) for td in (3.5, 6.0, 10.0, 16.0)]
    _, tr0, td0 = min(grid)
    best = minimize(cost, np.log([tr0, td0]), method="Nelder-Mead",
                    options={"xatol": 0.02, "fatol": 1e-3, "maxiter": 30})
    return tuple(np.exp(best.x))


def tau_pinned(tau_rise, tau_decay):
    """Whether either time constant ended up on a bound, i.e. was not actually identified."""
    return (min(abs(tau_rise - b) for b in cfg.TAU_RISE_RANGE) < 0.02,
            min(abs(tau_decay - b) for b in cfg.TAU_DECAY_RANGE) < 0.05)


def extract_events(u, min_amp=None, t0=0.0):
    """Collapse the raw solution into a list of impulses -> (times, amplitudes).

    FISTA spreads one physiological burst over two or three neighbouring samples, because it
    cannot tell which exact tick the burst started on.  Runs within cfg.MERGE_WINDOW_S are merged
    into a single event, taking the time of the largest sample and the sum of the run as the
    amplitude.

    t0 is the timestamp of the first sample.  Recordings do not all start at 0 (participant 99
    starts at 209 s), so without it every impulse would be labelled with the wrong block.
    """
    min_amp = cfg.MIN_EVENT_uS if min_amp is None else min_amp
    span = int(cfg.MERGE_WINDOW_S * cfg.DFS)

    times, amps, i = [], [], 0
    while i < len(u):
        if u[i] > 0:
            j = i
            while j + 1 < len(u) and u[j + 1] > 0 and (j + 1 - i) < span:
                j += 1
            total = u[i:j + 1].sum()
            if total >= min_amp:
                times.append(t0 + (i + int(np.argmax(u[i:j + 1]))) / cfg.DFS)
                amps.append(float(total))
            i = j + 1
        else:
            i += 1
    return np.array(times), np.array(amps)


def debias(phasic, h, ev_t, t0=0.0):
    """Re-estimate event amplitudes by non-negative least squares, times held fixed.

    The L1 penalty shrinks every impulse by roughly the same amount, so FISTA amplitudes are
    biased low.  Once the events are found the penalty has done its job: fitting one kernel per
    event to the phasic signal, with no penalty, removes the shrinkage.  Events are at least
    cfg.MERGE_WINDOW_S apart, so the columns are well separated and the fit is well posed.

    lsq_linear with bounds, not scipy.optimize.nnls: the pure-Python nnls in scipy 1.12-1.13
    overflows on this problem.
    """
    from scipy.optimize import lsq_linear
    n = len(phasic)
    idx = np.round((ev_t - t0) * cfg.DFS).astype(int).clip(0, n - 1)
    H = np.zeros((n, len(idx)))
    for j, k in enumerate(idx):
        H[k:k + len(h), j] = h[:n - k]
    amps = lsq_linear(H, phasic, bounds=(0, np.inf), method="bvls").x
    return amps, H @ amps


def burst_scale(phasic):
    """Typical burst size of a participant, measured before any deconvolution (uS).

    The median prominence of the phasic peaks that clear cfg.MIN_EVENT_uS.  Used to scale lam:
    deconvolving c*y with lam*c gives c times the solution for y with lam, so a lam proportional
    to the participant's own scale treats weak and strong responders identically.
    """
    from scipy.signal import find_peaks
    _, p = find_peaks(phasic, prominence=cfg.MIN_EVENT_uS)
    return float(np.median(p["prominences"])) if len(p["prominences"]) else np.nan


def gcv_score(seg, h, lam):
    """Generalized cross-validation score of one lam -> (score, number of active samples).

    GCV = n * ||seg - h*u||^2 / (n - df)^2.  For an L1 solution the degrees of freedom are the
    number of non-zero coefficients (Zou, Hastie & Tibshirani, Ann Stat 35:2173, 2007), here the
    non-zero samples of u before merging.  A small lam fits noise but spends degrees of freedom,
    a large lam leaves signal in the residual; the score balances the two.
    """
    u = deconvolve(seg, h, lam=lam)
    r = seg - np.convolve(u, h)[:len(seg)]
    n, df = len(seg), int((u > 0).sum())
    return n * (r @ r) / max(n - df, 1) ** 2, df


def fit_kernel_and_lambda(phasic, seg_s=300):
    """Alternate between the kernel fit and a GCV choice of lam, as in Faghih et al. (2014).

    The kernel is fitted at the current lam; lam is then re-chosen by GCV with that kernel, on the
    same segment the kernel fit uses.  Stops when lam no longer changes, or after
    cfg.GCV_ROUNDS rounds.  Returns (tau_rise, tau_decay, lam, rounds used).
    """
    seg = phasic[:int(seg_s * cfg.DFS)]
    lam = cfg.GCV_START
    for k in range(1, cfg.GCV_ROUNDS + 1):
        tau_r, tau_d = fit_kernel(phasic, seg_s=seg_s, lam=lam)
        h = kernel(tau_r, tau_d)
        new = min(cfg.GCV_LAMS, key=lambda l: gcv_score(seg, h, l)[0])
        if new == lam:
            break
        lam = new
    else:
        tau_r, tau_d = fit_kernel(phasic, seg_s=seg_s, lam=lam)
    return tau_r, tau_d, lam, k


def run(pid, load_fn):
    """The whole of stage 1 for one participant.

    Returns a dict with the downsampled signal, the tonic and phasic components, the fitted
    kernel, the impulse train, the reconstruction and the merged event list.
    """
    d = to_deconv_rate(load_fn(pid))
    tonic, phasic = split_tonic_phasic(d["eda"])
    if cfg.LAM_MODE == "gcv":
        tau_r, tau_d, lam, _ = fit_kernel_and_lambda(phasic)
    elif cfg.LAM_MODE == "scale":
        lam = cfg.LAM_SCALE_C * burst_scale(phasic)
        tau_r, tau_d = cfg.POOLED_KERNEL or fit_kernel(phasic, lam=lam)
    else:
        lam = cfg.LAM
        tau_r, tau_d = cfg.POOLED_KERNEL or fit_kernel(phasic)
    h = kernel(tau_r, tau_d)
    u = deconvolve(phasic, h, lam=lam)
    fit = np.convolve(u, h)[:len(phasic)]
    ev_t, ev_a = extract_events(u, t0=d["t"][0])
    if cfg.DEBIAS:
        ev_a, fit = debias(phasic, h, ev_t, t0=d["t"][0])

    return {**d, "tonic": tonic, "phasic": phasic, "h": h, "u": u, "fit": fit,
            "tau_rise": tau_r, "tau_decay": tau_d, "lam": lam, "ev_t": ev_t, "ev_a": ev_a,
            "r2": 1 - (phasic - fit).var() / phasic.var()}
