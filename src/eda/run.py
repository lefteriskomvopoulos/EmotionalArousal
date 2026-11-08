"""
Command-line entry point.  From the project root:

    PYTHONPATH=src python -m eda.run <command> [participant_id]

docs/EXPERIMENTS.md explains what each step found.  Commands in the order they are run.

Data
    quality            screen all 187 participants -> data/quality_study6.csv
    profile [pid]      model-free profile of every usable participant + their plain 2 Hz files
                       -> data/profile_study6*.csv/.npz, data/tbme_input/P*.mat
    synth              36 synthetic sessions drawn from the profiles, with their truth
                       -> data/tbme_synth_cohort.csv, data/tbme_synth_check.csv,
                          data/tbme_input/S*.mat, data/tbme_synth/S*.npz

The solver's input (own_preprocess.py)
    raw-atlas          all 119 participants + 18 tuning sessions, unprocessed, ~6 min
                       -> results/figures/raw_atlas_*.png, data/raw_overview.csv
    preprocess-stages  every preprocessing step, 5 real + 5 tuning sessions, ~2 min
                       -> results/figures/stages_*.png, data/preprocess_stages.csv
    own-check          all 119 through prepare: roughness, jumps, gaps + the filter's response, ~8 min
                       -> data/input_check.csv, data/filter_response.csv, results/figures/filter_response.png
    own-export         needs own-check.  0.5 Hz low-pass, 2 Hz: 36 synthetic + 5 real
                       -> data/tbme_input/*_own.mat, data/tbme_synth/*_own.npz
    own4-export        1 Hz low-pass, 4 Hz, noise matched to the sources at that rate (the current input)
                       -> data/input_check_own4.csv, data/tbme_input/*_own4.mat, data/tbme_synth/*_own4.npz

The cluster runs (src/own/make_own_bundle.sh, then src/own/run_own.sbatch on Torch)
    own-score          every setting in hpc/own/tune/output/<setting>, against the truth
                       -> data/own_tune_scores.csv
"""
import sys

from . import cohort_profile
from . import own_preprocess
from . import tbme_synth
from .quality import run_screen

COMMANDS = {
    "quality": lambda pid: run_screen(),
    "profile": lambda pid: cohort_profile.run(None if pid is None else [pid]),
    "synth": lambda pid: tbme_synth.run(),
    "raw-atlas": lambda pid: own_preprocess.atlas(),
    "preprocess-stages": lambda pid: own_preprocess.stages(),
    "own-check": lambda pid: own_preprocess.check_inputs(),
    "own-export": lambda pid: own_preprocess.export(),
    "own4-export": lambda pid: own_preprocess.export(suffix="own4"),
    "own-score": lambda pid: own_preprocess.score(),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    pid = int(argv[1]) if len(argv) > 1 else None
    COMMANDS[argv[0]](pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
