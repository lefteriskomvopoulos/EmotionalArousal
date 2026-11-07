"""
Paths and every tunable number in the pipeline, in one place.

Each value that was chosen rather than inherited carries the reason it was chosen, so that a
reader (or a reviewer) can see which numbers are physiology, which are literature, and which
are our own judgement.
"""
from pathlib import Path

# --------------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "Study6"              # the POPANE Study 6 csv files
META = ROOT / "docs" / "dataset_check" / "metadata_R1.xlsx"   # official metadata workbook
QUALITY_CSV = ROOT / "data" / "quality_study6.csv"  # written by the quality screen
KERNEL_CSV = ROOT / "data" / "kernels_study6.csv"   # written by the kernel-identifiability check
FIG_DIR = ROOT / "results" / "figures"

GROUPS = ("neutral", "negative", "positive", "mixed")

# --------------------------------------------------------------------------------- sampling
FS = 100       # Hz, rate used for loading and plotting (raw recordings are 1000 Hz)
DFS = 10       # Hz, rate used for deconvolution and state estimation.
               # An SCR rises over 1-4 s, so 10 samples/s resolves it with room to spare, and
               # EDA carries no content above ~1 Hz.  Higher rates cost time and add nothing.

# --------------------------------------------------------------------------------- screening
# Thresholds for rejecting a recording.  Only the amplitude criterion is derived from published
# guidance (Boucsein et al., Psychophysiology 49:1017, 2012); the rest are our own, chosen from
# the distribution across the 187 participants and fixed before any analysis was run.
MAX_SPIKE_FRAC = 0.03        # fraction of samples jumping > SPIKE_JUMP_uS -> electrode noise
SPIKE_JUMP_uS = 0.05         # an SCR rises over 1-4 s, so a jump this big in 0.1 s is an artefact
MAX_MISSING_FRAC = 0.05      # fraction of empty EDA samples
MIN_BURSTS_PER_MIN = 2       # a near-flat recording
MIN_MEDIAN_BURST_uS = 0.02   # electrodermal non-responders.  Recordings are quantised to
                             # 0.01 uS and the published "no response" cut is also 0.01 uS, so a
                             # participant whose MEDIAN burst is under twice that has most of
                             # their activity on the resolution floor.

# --------------------------------------------------------------------------------- tonic removal
# The drift is estimated as a rolling low percentile of the signal: bursts only push conductance
# upward, so the floor of the signal follows the drift beneath them.
TONIC_PCT = 3        # percentile defining the floor (1, 3, 5, 10 are all defensible)
TONIC_WIN_S = 30     # window over which the floor is measured.  At 60 s the fitted decay constant
                     # came out at ~2x published values, because slow content left in the signal
                     # is absorbed into a stretched kernel.
TONIC_SMOOTH_S = 30  # smoothing applied to the floor afterwards

# --------------------------------------------------------------------------------- deconvolution
KERNEL_S = 40        # kernel length in seconds (long enough to cover the full decay)
LAM = 1.6            # sparsity weight.  ONE global value for every participant, no per-subject
                     # scaling, so amplitudes stay comparable across people.  Chosen by sweeping
                     # it on the weakest and strongest responders in each group and keeping the
                     # value at which both fell in the physiological 5-15 impulses/min band.
MIN_EVENT_uS = 0.02  # impulses smaller than this are discarded (see MIN_MEDIAN_BURST_uS)
MERGE_WINDOW_S = 1.0 # FISTA spreads one burst over 2-3 samples; merge runs within this window

# Fixes under test on synthetic data (validation.compare_fixes).  Defaults keep the pipeline as
# presented in the proposal until the test picks a winner.
KERNEL_COST = "penalised"   # "penalised" (SSE + LAM*sum u) or "sse" (reconstruction error only)
DEBIAS = False              # refit event amplitudes by NNLS once their times are found

# Per-participant lambda by generalized cross-validation (Faghih et al., 2014).  On synthetic
# data the lambda that gives an unbiased kernel grew with burst size (0.03 uS -> <0.2,
# 0.06 uS -> ~0.3, 0.12 uS -> ~0.8), so one global value biases the kernel by responder type.
# Debiasing removes lambda's shrinkage of the amplitudes, so a per-participant lambda no longer
# costs amplitude comparability.
LAM_MODE = "global"         # "global" (LAM for everyone), "gcv", or "scale" (LAM_SCALE_C x
                            # the participant's burst scale, see deconvolution.burst_scale).
                            # GCV picked the smallest lambda on the grid whatever the burst
                            # size, so it is kept only for the record.
LAM_SCALE_C = 4.0           # provisional; set by validation.scale_test
GCV_LAMS = (0.07, 0.1, 0.14, 0.2, 0.28, 0.4, 0.57, 0.8, 1.13, 1.6, 2.26)   # x sqrt(2) steps
GCV_START = 0.4             # lambda for the first kernel fit
GCV_ROUNDS = 3              # kernel / lambda alternations at most

# Physiological bounds on the kernel time constants (seconds).  Without them the optimiser
# shrinks the kernel to a narrow spike, which fits anything (stack many along a decay) but is
# not a sweat-gland response.
TAU_RISE_RANGE = (0.5, 2.0)
TAU_DECAY_RANGE = (3.0, 20.0)

# Set to a (tau_rise, tau_decay) tuple to use ONE kernel for everybody instead of fitting one
# per participant.  Per-participant fitting is the default; pooling was tried when ~40% of
# participants had a tau pinned to a bound (see experiments.kernel_identifiability).
POOLED_KERNEL = None

# --------------------------------------------------------------------------------- state estimation
BIN_S = 5.0        # bin width for the state estimate.  Arousal varies over tens of seconds;
                   # 2 s bins made the state chase individual bursts.
EM_ITERS = 15      # expectation-maximization rounds

# The amplitude model r_k ~ N(g0 + g1*x_k, sr) has three parameters.  Learning all of them from a
# single recording is unstable: EM drives sr toward zero, every burst then looks like an exact
# measurement of arousal, g1 inflates and the state jitters.  So the amplitude noise is fixed and
# the slope is capped, and EM learns only the drift rate.  Both are provisional, pending the
# synthetic validation.
SR_FIXED = 0.5     # spread of log-amplitudes around the fitted line (natural log units)
G1_MAX = 1.0       # upper cap on how strongly arousal may scale burst size
G1_MIN = 0.0       # lower bound: arousal cannot make bursts SMALLER (physiology, not stability)

# --------------------------------------------------------------------------------- plotting
PLOT_PER_GROUP = 2   # how many usable participants per group the raw-signal figure shows
PLOT_IDS = None      # or an explicit list like [6, 12, 33] to override that choice

COLOR_SIGNAL = "#2a78d6"    # measured signal
COLOR_TONIC = "#eb6834"     # tonic / slider
COLOR_EVENTS = "#1baf7a"    # recovered impulses and estimated state

# --------------------------------------------------------------------------------- Amin & Faghih
# Events the published code returns below this size (its own units, u) are dropped afterwards.
# Chosen on half the synthetic sessions and checked on the other half (0.48 / 0.55 picked):
# held-out F1 +0.05, precision ~0.70 -> 0.80, recall ~0.79 -> 0.71 (tbme_score.cutoff_test).
# Raw events are always kept alongside.
TBME_CUT_U = 0.5
