"""
Noise: what the synthetic inputs get so they are as rough as the real ones, and how noise is
measured.  Shared by both generators (synth_v1, generator) and their checks.

The synthetic noise needs one correction.  White noise matched to the real signal at 100 Hz is
mostly removed by the low-pass, while real signals keep slow disturbances, so the synthetic input
came out about 3.5x smoother.  matched() adds slow noise until each session's 2 Hz roughness
equals its source participant's.  The true pulses do not change.

  matched        flat noise up to a target roughness on the solver's input (noise_2hz)
  colored_noise  Gaussian noise with a given slow spectrum (the slow-noise session variants)
  slow_deficit   the slow power flat sessions lack: real minus synthetic quiet-stretch spectrum
  spectra        full-input and quiet-stretch power spectra of a solver input; band_power

The spectrum tables are written by synth_v1.noise_spectrum (data/noise_spectrum*.csv, the first
18 tuning sessions) and generator.check50 (data/noise_spectrum50_curves.csv, the 50 selected).
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from . import config as cfg
from .preprocess import FIR_CUTOFF_HZ, FSY, low_pass, noise_2hz, prepare

MATCH_SEED = 2026       # matched()'s default noise seed is this + the session's number: the first
                        # generator's sessions (S, F, T); the same number as synth_v1.SPLIT_SEED
SLOW_BAND_HZ = (0.03, 0.6)     # where the real quiet stretches carry more power than the tuning sessions
SPECTRUM50_CURVES_CSV = cfg.ROOT / "data" / "noise_spectrum50_curves.csv"
SPECTRUM_CSV = cfg.ROOT / "data" / "noise_spectrum.csv"
SPECTRUM_CURVES_CSV = cfg.ROOT / "data" / "noise_spectrum_curves.csv"
BANDS_HZ = ((0.06, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0))
QUIET_S, QUIET_BEFORE_S, QUIET_AFTER_S = 16, 5, 3


def colored_noise(n, fs, f_shape, psd, rng, band=SLOW_BAND_HZ):
    """Gaussian noise at rate fs whose one-sided power spectrum is psd(f_shape) (uS^2/Hz) inside
    band and zero outside.  Absolute units: unit white noise has density 2/fs, so the gain is
    sqrt(psd fs / 2).  Below the first measured frequency the spectrum is held at that bin."""
    p = np.asarray(psd, float).copy()
    p[0] = p[1]                                          # the detrended 0 Hz bin holds nothing
    f = np.fft.rfftfreq(n, 1 / fs)
    gain = np.sqrt(np.interp(f, f_shape, p, right=0.0) * fs / 2)
    gain[(f < band[0]) | (f > band[1])] = 0.0
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * gain, n)


def slow_deficit(curves_csv=None):
    """The slow noise flat-noise sessions lack: real minus synthetic quiet-stretch spectrum
    (median over pairs), floored at 0.  -> (f, psd in uS^2/Hz).
    curves_csv: SPECTRUM_CURVES_CSV (noise-spectrum, the first 18 tuning sessions; used by the
    first test set) or SPECTRUM50_CURVES_CSV (synth-check, the 50 selected; round 6 on)."""
    cur = pd.read_csv(curves_csv or SPECTRUM_CURVES_CSV).query("kind == 'quiet'").sort_values("f_hz")
    d = (cur.psd_real - cur.psd_synthetic).clip(lower=0).to_numpy()
    if not np.isfinite(d).all():
        raise RuntimeError("quiet spectrum has gaps: re-run noise-spectrum")
    return cur.f_hz.to_numpy(), d


def matched(name, row, s, t, targets, cutoff=FIR_CUTOFF_HZ, fsy=FSY, seed=None):
    """The session's 100 Hz signal plus slow noise, so its roughness at fsy equals its source's.
    The noise seed is MATCH_SEED + the session's number unless seed is given."""
    base = noise_2hz(prepare(s["x"], t, cfg.FS, (0, len(t) - 1), cutoff, fsy)["y"])
    extra = np.sqrt(max(targets[int(row.id)] ** 2 - base ** 2, 0.0))
    if extra == 0:
        return s["x"]
    seed = MATCH_SEED + int(name[1:]) if seed is None else seed
    w = low_pass(np.random.default_rng(seed).standard_normal(len(t)), cfg.FS, cutoff)
    return s["x"] + w * extra / noise_2hz(w[::int(cfg.FS // fsy)])


def spectra(y, fs):
    """Full-input and quiet-stretch power spectra of one solver input (detrended, Hann).

    Quiet: 16 s windows (half-overlapping) with no peak of prominence >= 0.02 uS from 5 s before to
    3 s after, so no counted response rises or peaks inside.  Tiny responses below that stay in,
    in real and synthetic alike.
    """
    from scipy.signal import welch
    f_full, p_full = welch(y, fs=fs, nperseg=int(64 * fs), detrend="linear")
    n, step = int(QUIET_S * fs), int(QUIET_S * fs / 2)
    peaks, _ = find_peaks(y, prominence=cfg.MIN_EVENT_uS)
    psd = []
    for a in range(0, len(y) - n + 1, step):
        lo, hi = a - QUIET_BEFORE_S * fs, a + n + QUIET_AFTER_S * fs
        if not ((peaks >= lo) & (peaks < hi)).any():
            psd.append(welch(y[a:a + n], fs=fs, nperseg=n, detrend="linear")[1])
    f_q = np.fft.rfftfreq(n, 1 / fs)
    p_quiet = np.mean(psd, axis=0) if psd else np.full(len(f_q), np.nan)
    return {"full": (f_full, p_full), "quiet": (f_q, p_quiet)}, len(psd)


def band_power(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(p[m].sum() * (f[1] - f[0]))
