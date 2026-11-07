"""
Synthetic skin conductance with a known answer, for validating stage 1.

The signal is built from the same model the deconvolution assumes,

    y[k] = sum_i a_i h(t_k - t_i) + tonic(t_k) + noise[k],     rounded to 0.01 uS

where h is the peak-normalised kernel from deconvolution.kernel, i.e. the impulse response of
the second-order model of Amin & Faghih (IEEE TBME, 2021).  Because the kernel peaks at 1, each
a_i is the peak height of its response in uS.  Every impulse, both time constants and the drift
are known, so recovery can be scored exactly.

The signal is generated directly at cfg.DFS: the anti-aliasing step is not what is under test.
"""
import numpy as np

from .. import config as cfg
from .deconvolution import kernel

QUANT_uS = 0.01   # resolution of the POPANE recordings


def impulses(duration_s, rate_pm, amp_median_uS, amp_sigma, rng):
    """Impulse times from a renewal process, amplitudes from a log-normal -> (times, amps).

    Gaps are an exponential shifted by cfg.MERGE_WINDOW_S, so the mean rate is still rate_pm.
    Two true impulses closer than that window would count as one event under our own
    definition (deconvolution.extract_events), which would make the scoring unfair.
    """
    gap_min = cfg.MERGE_WINDOW_S
    mean_gap = 60.0 / rate_pm
    times, t = [], rng.exponential(mean_gap)
    while t < duration_s:
        times.append(t)
        t += gap_min + rng.exponential(mean_gap - gap_min)
    times = np.array(times)
    amps = amp_median_uS * np.exp(amp_sigma * rng.standard_normal(len(times)))
    return times, amps


def tonic_drift(n, level_uS, drift_uS, rng, smooth_s=60):
    """Slow baseline: a random walk smoothed over smooth_s, wandering about drift_uS in total."""
    from scipy.ndimage import uniform_filter1d
    walk = np.cumsum(rng.standard_normal(n)) * drift_uS / np.sqrt(n)
    return level_uS + uniform_filter1d(walk, size=int(smooth_s * cfg.DFS))


def modulate(ev_t, ev_a, duration_s, depth, rng, period_s=240):
    """Thin the impulse train with a slow random rate profile, mimicking arousal changes.

    Each impulse survives with probability p(t) in [1 - depth, 1], p following a smoothed random
    curve with a time scale of period_s.  The rate therefore varies within the session instead
    of being constant, which is the situation the state estimator exists for.
    """
    knots = rng.uniform(size=int(duration_s / period_s) + 2)
    p = 1 - depth + depth * np.interp(ev_t, np.linspace(0, duration_s, len(knots)), knots)
    keep = rng.uniform(size=len(ev_t)) < p
    return ev_t[keep], ev_a[keep]


def coloured_noise(n, sd_uS, ar, rng):
    """AR(1) noise with stationary standard deviation sd_uS; ar = 0 gives white noise."""
    from scipy.signal import lfilter
    e = rng.standard_normal(n) * sd_uS * np.sqrt(1 - ar ** 2)
    return lfilter([1.0], [1.0, -ar], e)


def artefacts(n, per_min, rng, amp_range=(0.05, 0.2), dur_range_s=(0.1, 0.5)):
    """Brief electrode jumps: rectangular pulses of either sign, too fast to be sweat."""
    x = np.zeros(n)
    for _ in range(rng.poisson(per_min * n / cfg.DFS / 60)):
        k = rng.integers(n)
        x[k:k + int(rng.uniform(*dur_range_s) * cfg.DFS) + 1] += \
            rng.choice((-1, 1)) * rng.uniform(*amp_range)
    return x


def simulate(duration_s=1200, tau_rise=1.0, tau_decay=6.6, rate_pm=10, amp_median_uS=0.06,
             amp_sigma=0.6, level_uS=2.0, drift_uS=0.5, noise_uS=0.01, noise_ar=0.0,
             rate_depth=0.0, artefacts_pm=0.0, seed=0):
    """One synthetic session at cfg.DFS Hz.

    Defaults follow what the real cohort gave us: tau_decay 6.6 s is the cohort median,
    10 impulses/min sits in the 5-15 band used to choose lambda, and noise_uS matches the
    ~0.01 uS high-frequency jitter measured by the quality screen.  The defaults give the easy
    case (white noise, constant rate, no artefacts); noise_ar, rate_depth and artefacts_pm make
    it harder, see realistic_cohort.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * cfg.DFS)
    t = np.arange(n) / cfg.DFS

    ev_t, ev_a = impulses(duration_s, rate_pm, amp_median_uS, amp_sigma, rng)
    if rate_depth > 0:
        ev_t, ev_a = modulate(ev_t, ev_a, duration_s, rate_depth, rng)
    u = np.zeros(n)
    np.add.at(u, np.round(ev_t * cfg.DFS).astype(int).clip(0, n - 1), ev_a)

    phasic = np.convolve(u, kernel(tau_rise, tau_decay))[:n]
    tonic = tonic_drift(n, level_uS, drift_uS, rng)
    eda = phasic + tonic + coloured_noise(n, noise_uS, noise_ar, rng)
    if artefacts_pm > 0:
        eda = eda + artefacts(n, artefacts_pm, rng)
    eda = np.round(eda / QUANT_uS) * QUANT_uS

    return {"t": t, "eda": eda, "phasic": phasic, "tonic": tonic, "u": u,
            "ev_t": ev_t, "ev_a": ev_a, "tau_rise": tau_rise, "tau_decay": tau_decay,
            "amp_median_uS": amp_median_uS}


def cohort(n_subjects=20, seed=0, **kw):
    """Several sessions with time constants drawn inside the physiological bounds.

    The ranges stay away from the bounds in config.TAU_*_RANGE on purpose: a fit that lands
    on a bound when the truth is in the interior is exactly the failure we want to detect.
    """
    rng = np.random.default_rng(seed)
    return [simulate(tau_rise=rng.uniform(0.7, 1.5), tau_decay=rng.uniform(4.0, 12.0),
                     seed=seed + 1 + i, **kw)
            for i in range(n_subjects)]


def realistic_cohort(n_subjects=20, seed=100):
    """Harder sessions, every property drawn per session instead of fixed.

    Burst size 0.02-0.15 uS (log-uniform, the span of our cohort's median bursts), mean rate
    4-16/min modulated by up to 70% within the session, correlated noise (AR 0.8-0.95, i.e. a
    correlation time of 0.5-2 s), and 0-0.5 electrode artefacts per minute.  The seed differs
    from cohort() so nothing here was seen while choosing settings.
    """
    rng = np.random.default_rng(seed)
    return [simulate(tau_rise=rng.uniform(0.7, 1.5), tau_decay=rng.uniform(4.0, 12.0),
                     amp_median_uS=float(np.exp(rng.uniform(np.log(0.02), np.log(0.15)))),
                     rate_pm=rng.uniform(4, 16), rate_depth=rng.uniform(0.3, 0.7),
                     noise_uS=rng.uniform(0.008, 0.017), noise_ar=rng.uniform(0.8, 0.95),
                     artefacts_pm=rng.uniform(0, 0.5), seed=seed + 1 + i)
            for i in range(n_subjects)]
