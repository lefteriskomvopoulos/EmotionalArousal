"""
Reading the POPANE Study 6 files.

Each participant has one file, S6_P<id>_All.csv, recorded at 1000 Hz with columns:

    timestamp | affect (1-10 slider) | ECG | EDA (uS) | SBP DBP CO TPR | marker

`marker` is a per-sample block code: -1 is the five-minute resting baseline, 1xx a neutral film,
2xx a negative film, 3xx a positive film, and an empty value is unlabelled time.  Recordings do
not all start at t = 0; participant 99, for instance, starts at 209 s.
"""
import numpy as np
import pandas as pd

from . import config as cfg


def read_header(path):
    """The '#' comment lines at the top of a POPANE csv -> {key: [values]}.

    These carry the channel names, units, sampling rates and recording devices.
    """
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            key, *vals = line.strip().lstrip("#").split(",")
            meta[key] = vals
    return meta


def load(pid, every=None):
    """One participant -> dict of numpy arrays at cfg.FS Hz.

    Keys: t (seconds), eda (uS), affect (1-10 slider), marker (block code as a string).
    `every` is the decimation factor applied to the 1000 Hz file; it simply keeps every n-th
    sample, which is fine for plotting.  Analysis uses signal.to_deconv_rate instead, which
    low-pass filters first.
    """
    every = int(1000 // cfg.FS) if every is None else every
    df = pd.read_csv(cfg.RAW / f"S6_P{pid}_All.csv", comment="#",
                     usecols=["timestamp", "EDA", "affect", "marker"],
                     dtype={"marker": str}, keep_default_na=False, na_values=[""])
    df = df.iloc[::every]
    return {"t": df["timestamp"].to_numpy(float),
            "eda": df["EDA"].to_numpy(float),
            "affect": df["affect"].to_numpy(float),
            "marker": df["marker"].fillna("none").to_numpy(str)}


def groups_from_metadata():
    """{participant_id: (group, eda_flagged)} read from the official metadata workbook.

    The group follows from the six film IDs the participant watched: all 1xx is neutral, all 2xx
    negative, all 3xx positive, and any mixture is the mixed group.  `eda_flagged` is True when
    the dataset authors themselves marked this participant's EDA as missing, low SNR or
    artefactual.
    """
    import openpyxl
    ws = openpyxl.load_workbook(cfg.META, read_only=True, data_only=True)["study6"]
    rows = [r for r in ws.iter_rows(values_only=True)
            if r[0] is not None and not str(r[0]).startswith("#")]
    hdr, rows = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(hdr)}

    out = {}
    for r in rows:
        families = {str(r[col[f"stimuli{i}"]])[0] for i in range(1, 7)}
        group = ({"1": "neutral", "2": "negative", "3": "positive"}[families.pop()]
                 if len(families) == 1 else "mixed")
        flags = " ".join(str(r[col[c]]) for c in ("missing", "SNR outlier", "Artefacts"))
        out[r[col["ID"]]] = (group, "EDA" in flags or flags.strip() == "all")
    return out


# --------------------------------------------------------------------------- block masks
def baseline_mask(marker):
    """True where the participant was in the five-minute resting baseline."""
    return marker == "-1"


def film_mask(marker):
    """True where any film was playing (block codes 1xx, 2xx, 3xx)."""
    return np.isin(np.char.ljust(marker.astype(str), 1), ["1", "2", "3"])


def block_changes(marker):
    """Indices where the block code changes, for drawing boundaries on a figure."""
    return np.flatnonzero(marker[1:] != marker[:-1]) + 1
