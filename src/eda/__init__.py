"""
POPANE Study 6 — decoding sympathetic arousal from skin conductance.

Two stages:
    src/tbme/ + tbme_*.py   skin conductance -> impulses, Amin & Faghih (2021) published solver
    state.py                impulses -> hidden arousal state, marked point process filter

archive/ holds the superseded deconvolution and cvxEDA code.  Everything tunable lives in config.py.
"""
__all__ = ["config", "dataset", "quality", "state", "figures"]
