"""
POPANE Study 6 — decoding sympathetic arousal from skin conductance.  Branch `own`.

Skin conductance -> impulses with our own implementation of Amin & Faghih (IEEE TBME 2021)
(src/own/, MATLAB), tuned on synthetic sessions with a known truth.  This package prepares its
input, builds the synthetic sessions, and scores the output.  One module per stage, in run order:

    quality, dataset, config   screen and load the recordings
    cohort_profile             model-free profile of every usable participant
    preprocess                 the solver's input: every step in one order (prepare), its checks
    noise                      noise matched to roughness, slow noise, spectra
    tbme_synth, synth_v1       the first generator and its runs (tuning rounds 1-5, first final run)
    selection                  the 50 participants analysed (baseline-only rule, seeded draw)
    generator                  the frozen generator from round 6 on (tuning U, test V)
    tuning                     scoring every tuning round, and its diagnostics
    runs                       the final run, re-synthesis, figures, the real pulses for stage 2
    scr_shapes                 model-free SCR shapes on the waveforms (a description; not a prior)
    tbme_export, tbme_score    input files in the solver's format; scoring against the truth
    figures                    every figure
    own_preprocess             compatibility only: the old single module, re-exporting the above

Entry point: run.py (PYTHONPATH=src python -m eda.run, lists every command).
"""
__all__ = ["config", "dataset", "quality", "cohort_profile", "preprocess", "noise", "tbme_synth",
           "synth_v1", "selection", "generator", "tuning", "runs", "scr_shapes", "tbme_export",
           "tbme_score", "figures"]
