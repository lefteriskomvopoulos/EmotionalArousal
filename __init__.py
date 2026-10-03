"""
POPANE Study 6 — decoding sympathetic arousal from skin conductance.

Two stages:
    deconvolution.py   skin conductance -> sudomotor impulses (times and amplitudes)
    state.py           impulses -> hidden arousal state, by marked point process filtering

Everything tunable lives in config.py.
"""
__all__ = ["config", "dataset", "quality", "deconvolution", "state", "figures", "experiments"]
