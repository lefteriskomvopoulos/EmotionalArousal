"""
Command-line entry point.  From the project root:

    PYTHONPATH=src python -m eda.run <command> [participant_id]

docs/EXPERIMENTS.md explains what each step found.  Commands in the order they are run.
Module per section in brackets.  Only the last command is needed to start stage 2: it reads the
final run's output, which is in the repository.

Data (quality.py, cohort_profile.py, tbme_synth.py)
    quality            screen all 187 participants -> data/quality_study6.csv
    profile [pid]      model-free profile of every usable participant + their plain 2 Hz files
                       -> data/profile_study6*.csv/.npz, data/tbme_input/P*.mat
    synth              the first generator: 36 synthetic sessions drawn from the profiles, with truth
                       -> data/tbme_synth_cohort.csv, data/tbme_synth_check.csv,
                          data/tbme_input/S*.mat, data/tbme_synth/S*.npz

The solver's input (preprocess.py)
    raw-atlas          all 119 participants + 18 tuning sessions, unprocessed, ~6 min
                       -> results/figures/raw_atlas_*.png, data/raw_overview.csv
    preprocess-stages  every preprocessing step, 5 real + 5 tuning sessions, ~2 min
                       -> results/figures/stages_*.png, data/preprocess_stages.csv
    preprocess-stages-own4  the same figure for P53 through the final chain (1 Hz low-pass, 4 Hz), ~1 min
                       -> results/figures/stages_P53_own4.png
    own-check          all 119 through prepare: roughness, jumps, gaps + the filter's response, ~8 min
                       -> data/input_check.csv, data/filter_response.csv, results/figures/filter_response.png

Tuning rounds 1-5 on the first generator (synth_v1.py, tuning.py; cluster: src/own/make_own_bundle.sh,
then src/own/run_own.sbatch on Torch)
    own-export         needs own-check.  0.5 Hz low-pass, 2 Hz: 36 synthetic + 5 real
                       -> data/tbme_input/*_own.mat, data/tbme_synth/*_own.npz
    own4-export        1 Hz low-pass, 4 Hz, noise matched to the sources at that rate (the current input)
                       -> data/input_check_own4.csv, data/tbme_input/*_own4.mat, data/tbme_synth/*_own4.npz
    own-score          every setting in hpc/own/tune/output/<setting>, against the truth (every
                       round, 1-7: S sessions for 1-5, U from 6)
                       -> data/own_tune_scores.csv
    own-decay-check    needs own-score.  Which sessions get too-long decays (best setting so far)
                       -> data/own_decay_check.csv, data/own_decay_sessions.csv
    own-errors         every miss and false pulse of the best setting, by size and closeness
                       -> data/own_errors.csv, data/own_errors_summary.csv
    noise-spectrum     each tuning session's input vs its source participant's, band by band, ~2 min
                       -> data/noise_spectrum.csv, data/noise_spectrum_curves.csv

Participant selection (selection.py): stage 1 = quality, stage 2 = baseline-only, stage 3 = draw
    select-measures    stage-2 measures of the 119, spread per group, ~8 min
                       -> data/selection_measures.csv
    select             stage 2 (median baseline SCR >= 0.05 uS) + the draw: the 50 (20/10/10/10)
                       -> data/selection.csv, data/selection_balance.csv
    own-score-selected the tuning scores over sessions whose source passes stage 2 only
                       -> data/own_tune_scores_selected.csv

Round 6 on: the frozen generator (generator.py, tuning.py; cluster: make_own_bundle.sh; scored by
own-score)
    synth-check        needs final-export.  How well the round-6
                       generator (each participant's own counted responses reused per run, one
                       physiological shape at random, noise matched; nothing tuned) fits the 50,
                       and the slow noise its flat sessions lack, ~15 min
                       -> data/synth50_check.csv, data/noise_spectrum50_curves.csv
    tune50-export      needs synth-check.  18 new tuning sessions built from the 50 selected
                       (6 neutral, 4 positive, 4 negative, 4 mixed), flat and slow noise, own seeds
                       -> data/tbme_input/U*_own4.mat, U*_own4s.mat, data/tbme_synth/U*.npz,
                          data/tune50_cohort.csv
    tune50-plots       needs tune50-export.  Each tuning session next to its real participant, ~1 min
                       -> results/figures/tune50_vs_real.pdf, results/figures/tune50/*.png
    tune50-preview     needs synth-check.  The sessions tune50-export would write, only plotted;
                       nothing else is saved, ~2 min
                       -> results/figures/tune50_preview_vs_real.pdf, results/figures/tune50_preview/*.png
    own-scale          needs tune50-export.  Each input's typical response size and gain
                       -> data/own_scale.csv
    round-compare      needs a round's output in hpc/own/tune/output (control and variants:
                       tuning.COMPARE_*).  Each variant against the control on the 18 U sessions,
                       session by session: pulses found / missed / false by size and closeness, the
                       tonic's share of the responses, the round's rule; one page per session with
                       every setting's tonic and pulses, ~3 min
                       -> data/round<k>_compare.csv, results/figures/round<k>_tonic.pdf (+ PNG folder)

The first final run (frozen own4_p1000_k14_f3; synth_v1.py; src/own/make_own_final_bundle.sh)
    final-export       needs select + noise-spectrum.  The 50 real inputs (P*_own4, the real input
                       of every later run too) and two test sessions per participant from the
                       first generator: F (flat noise, as tuning) and T (plus slow noise), ~10 min
                       -> data/tbme_input/P*_own4.mat, F*_own4.mat, T*_own4.mat,
                          data/tbme_synth/F*_own4.npz, T*_own4.npz, data/tbme_test50_cohort.csv
    final-score        the 100 test sessions in hpc/own/final/output/own4_p1000_k14_f3, once
                       -> data/own_final_test_scores.csv
                       (the first final run, round-5 setting; kept as run)

The final run after round 6 (frozen own4_p1000_k14_f3_s8; generator.py, runs.py;
src/own/make_own_final_bundle.sh: P + V)
    test50-export      needs final-export and synth-check.  The final test set: one session per
                       participant (V<id>) by the frozen generator, own seed, ~5 min
                       -> data/tbme_input/V*_own4.mat, data/tbme_synth/V*_own4.npz,
                          data/test50_cohort.csv
    test50-score       the 50 test sessions in hpc/own/final/output/own4_p1000_k14_f3_s8, once
                       -> data/own_test50_scores.csv

The papers' re-synthesis test and the figures (runs.py; src/own/make_own_resynth_bundle.sh)
    resynth-export     needs the final run's real outputs.  Each real signal rebuilt from our
                       deconvolution + white noise at their own fit residual (Amin & Faghih's method)
                       -> data/tbme_input/R*_own4.mat, data/tbme_synth/R*_own4.npz,
                          data/resynth50_cohort.csv
    resynth-score      the 50 rebuilt signals in hpc/own/resynth/output/own4_p1000_k14_f3_s8
                       -> data/own_resynth50_scores.csv
    final-plots        needs test50-score.  Every final-run signal before and after deconvolution:
                       test sessions with found / missed / false pulses, real ones with their pulses
                       -> results/figures/test50_deconv.pdf, real50_deconv.pdf (+ PNG folders)
    resynth-plots      needs resynth-score.  The rebuilt signals, as final-plots
                       -> results/figures/resynth50_deconv.pdf (+ PNG folder)

Model-free SCR shapes (scr_shapes.py)
    scr-shapes         needs the final run (P, for the comparison) and test50-export.  Half-rise and
                       half-recovery times of isolated SCRs on the 50 real recordings, as tau; per
                       participant against the solver's fits; checked on the 50 test sessions with a
                       pass criterion set beforehand, ~15 min
                       -> data/scr_shapes_real50.csv, prior_measured50.csv, scr_shapes_validation.csv
    pulse-check        needs scr-shapes and real-pulses.  How many pulses the frozen setting places under
                       each isolated SCR (on its rise, on its recovery), real 50 against the 50 test
                       sessions, whose true pulses are counted the same way, ~8 min
                       -> data/pulse_check_isolated.csv

Stage 2's input (runs.py)
    real-pulses        needs the final run's real outputs (hpc/own/final/output/own4_p1000_k14_f3_s8,
                       in the repository).  Every pulse of the 50 real participants with its time,
                       size in uS, block and film valence; and every window's fit, ~1 min
                       -> data/real50_pulses.csv, data/real50_windows.csv
    block-rates        needs real-pulses.  Descriptive pulse rates and sizes by block (baseline, neutral /
                       negative / positive films) and film group, >= 0.05 uS and all; no test.  A
                       first look at what H1-H3 are about, after stage 1 and the threshold were frozen
                       -> data/real50_block_rates.csv
"""
import sys

from . import cohort_profile
from . import generator
from . import preprocess
from . import runs
from . import scr_shapes
from . import selection
from . import synth_v1
from . import tbme_synth
from . import tuning
from .quality import run_screen

COMMANDS = {
    "quality": lambda pid: run_screen(),
    "profile": lambda pid: cohort_profile.run(None if pid is None else [pid]),
    "synth": lambda pid: tbme_synth.run(),
    "raw-atlas": lambda pid: preprocess.atlas(),
    "preprocess-stages": lambda pid: preprocess.stages(),
    "preprocess-stages-own4": lambda pid: preprocess.stages_variant(),
    "own-check": lambda pid: preprocess.check_inputs(),
    "own-export": lambda pid: synth_v1.export(),
    "own4-export": lambda pid: synth_v1.export(suffix="own4"),
    "own-score": lambda pid: tuning.score(),
    "own-decay-check": lambda pid: tuning.decay_check(),
    "own-errors": lambda pid: tuning.error_check(),
    "noise-spectrum": lambda pid: synth_v1.noise_spectrum(),
    "select-measures": lambda pid: selection.measures(),
    "select": lambda pid: selection.select(),
    "own-score-selected": lambda pid: selection.score_selected(),
    "synth-check": lambda pid: generator.check50(),
    "tune50-export": lambda pid: generator.export_tune50(),
    "tune50-plots": lambda pid: generator.plot_tune50(),
    "tune50-preview": lambda pid: generator.export_tune50(save=False),
    "own-scale": lambda pid: tuning.scale(),
    "round-compare": lambda pid: tuning.compare_round(),
    "final-export": lambda pid: synth_v1.export_selected(),
    "final-score": lambda pid: synth_v1.score_final(),
    "test50-export": lambda pid: generator.export_test50(),
    "test50-score": lambda pid: runs.score_test50(),
    "resynth-export": lambda pid: runs.export_resynth(),
    "resynth-score": lambda pid: runs.score_resynth(),
    "final-plots": lambda pid: runs.plot_final(),
    "resynth-plots": lambda pid: runs.plot_resynth(),
    "scr-shapes": lambda pid: scr_shapes.run(),
    "pulse-check": lambda pid: scr_shapes.pulse_check(),
    "real-pulses": lambda pid: runs.export_pulses(),
    "block-rates": lambda pid: runs.block_rates(),
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
