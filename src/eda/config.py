"""
Paths and the shared numbers of the pipeline, in one place.

Each value that was chosen rather than inherited carries the reason it was chosen, so that a
reader (or a reviewer) can see which numbers are physiology, which are literature, and which
are our own judgement.  The deconvolution's own settings live in src/own/own_config.m.
"""
from pathlib import Path

# --------------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "Study6"              # the POPANE Study 6 csv files
META = ROOT / "docs" / "dataset_check" / "metadata_R1.xlsx"   # official metadata workbook
QUALITY_CSV = ROOT / "data" / "quality_study6.csv"  # written by the quality screen
FIG_DIR = ROOT / "results" / "figures"

# --------------------------------------------------------------------------------- sampling
FS = 100       # Hz, rate used for loading, profiling and the synthetic sessions (raw is 1000 Hz)

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

# --------------------------------------------------------------------------------- events
MIN_EVENT_uS = 0.02  # the smallest SCR counted (twice the 0.01 uS resolution): the profile's
                     # peak threshold and the "visible" pulses of the scoring
MERGE_WINDOW_S = 1.0 # matching tolerance: a recovered pulse within 1 s of a true one is a hit

# --------------------------------------------------------------------------------- plotting
COLOR_SIGNAL = "#2a78d6"    # measured signal
COLOR_TONIC = "#eb6834"     # tonic / what a step changed
COLOR_EVENTS = "#1baf7a"    # samples the solver receives
