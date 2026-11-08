"""
POPANE Study 6 — decoding sympathetic arousal from skin conductance.  Branch `own`.

Skin conductance -> impulses with our own implementation of Amin & Faghih (IEEE TBME 2021)
(src/own/, MATLAB), tuned on synthetic sessions with a known truth.  This package prepares its
input and scores its output:

    quality, dataset, config   screen and load the recordings
    cohort_profile             model-free profile of every usable participant
    tbme_synth                 synthetic sessions drawn from those profiles (the known truth)
    own_preprocess             the solver's input, every step in one order; the tuning scores
    tbme_export, tbme_score    input files in the solver's format; scoring against the truth
    figures                    every figure
"""
__all__ = ["config", "dataset", "quality", "cohort_profile", "tbme_synth", "own_preprocess",
           "tbme_export", "tbme_score", "figures"]
