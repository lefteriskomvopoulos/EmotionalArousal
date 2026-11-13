"""
Compatibility only.  own_preprocess was split into one module per stage (2026-11-13), with no
change in behaviour; old names still resolve here.  New code imports the modules directly:

  preprocess   the solver's input: prepare() and its checks and figures
  noise        noise matched to roughness, slow noise, spectra
  synth_v1     the first generator, tuning rounds 1-5, the first final run (S, F, T)
  generator    the frozen generator from round 6 on (U, V)
  tuning       own-score, own-scale and the tuning diagnostics
  runs         the final run (V, P), re-synthesis (R), their figures, the real pulses for stage 2
"""
from .generator import *  # noqa: F401,F403
from .generator import _plot_sessions, _ratio  # noqa: F401
from .noise import *  # noqa: F401,F403
from .preprocess import *  # noqa: F401,F403
from .runs import *  # noqa: F401,F403
from .runs import _deconv_item  # noqa: F401
from .synth_v1 import *  # noqa: F401,F403
from .tbme_score import event_heights  # noqa: F401
from .tuning import *  # noqa: F401,F403
