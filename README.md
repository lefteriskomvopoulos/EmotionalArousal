# Decoding sympathetic arousal from skin conductance

Course project for BE-GY 9603 / ECE-GY 9153 A, Neural and Physiological Signal Processing.
Komvopoulos Eleftherios (ek4538) and Piyush More (pm4081).

The pipeline: skin conductance of POPANE Study 6 → **sudomotor impulses** (stage 1, our own
implementation of Amin & Faghih, IEEE TBME 2021) → **arousal state** (stage 2) → hypotheses H1–H6.
Both stages are done. Stage 2 starts from one table (below).

## Start here

1. **[REPRODUCE.md](REPRODUCE.md)**: one command, `./reproduce.sh`, recomputes the results and
   compares them with the shipped ones.
2. This README: what exists, where, and how to run it.
3. **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**: the whole story in order. Every experiment, its
   result, and the decision it justified, with the file each number comes from.

## What this folder combines

It is the `submission` branch of the repository, which joins the work of three branches:

| Branch | What it holds | Here |
|---|---|---|
| `deconvolution` | Stage 1: **our own implementation** of the Amin & Faghih (2021) algorithm, with every setting under our control, tuned on synthetic sessions with a known truth, then run on the real participants. | `src/eda/`, `src/own/`, `hpc/own/` |
| `stateestimation` | Stage 2: the arousal state estimated from the pulses, and the H1–H6 tests. | `src/stage2/`, `data/stage2_*` |
| `paper_solver` | Everything up to and including the published solver, run unmodified (encrypted MATLAB), the cvxEDA comparison, and the first stage-2 code (`state.py`). Frozen. | `archive/paper_solver/` (a record, not used by stages 1 or 2) |

We wrote our own implementation because the black box's settings could not be changed: its
response shape always stayed at its built-in prior. The algorithm is the published one; its
settings were tuned on our data.

## Where things stand

- **Frozen setting `own4_p1000_k14_f3_s8`** (`src/own/make_own_final_bundle.sh`). On 50 unseen
  synthetic test sessions built from the 50 analysed participants: F1 0.82, recall 0.85,
  precision 0.85, timing within 0.16 s, size rank correlation 0.98. For pulses ≥ 0.05 µS: F1 0.96,
  recall 0.98, precision 0.96 (nearly all errors are 0.02–0.05 µS pulses, at the noise floor).
- **50 real participants deconvolved** (20 neutral, 10 positive, 10 negative, 10 mixed film
  groups; `data/selection.csv`). Their pulses: `data/real50_pulses.csv` (`real-pulses`, below).
- **Known limits:** slow responses (half-recovery > 7 s) lose precision (0.74) and their decay
  comes out 37% short; on fast and typical shapes, which most real participants have, decays run
  long. The full list is in docs/EXPERIMENTS.md, Current status.
- **Round 7** (no prior against the frozen setting): no better and less stable, so the frozen
  setting stays. **Round 8** (stiffer tonic, knots every 20 or 28 s): fewer misses, more false
  pulses, no better on slow responses, so the frozen setting stays. **Re-synthesis** (the papers'
  test), all 50: F1 0.86, the same in every film group.
- **Real pulse counts are plausible** for a deconvolution, about 20% high (small doublets on response
  rises; docs/EXPERIMENTS.md Part 12).
- **Stage 2 and H1–H6 done** (docs/EXPERIMENTS.md Parts 13–14). The Data Challenge 2 filters were
  validated on synthetic sessions, then run on the 50. The tests were set in writing first, then
  run once.
  - **H2 supported.**
  - **H1, H3 and H4 not supported.** Within the mixed group, negative and positive films are
    equivalent.
  - **H5:** no evidence of separation.
  - **H6 not supported** (r +0.14, p 0.074; Part 19).
- **The affect slider** (Part 19, the evaluation of Oliver et al. 2025, adapted):
  - the state does not pick out the moments of highest self-reported intensity;
  - it follows the direction of changes slightly above chance (54–59%), the level the paper
    reports.
  - **Input:** `data/stage2_slider_1hz.csv`, built from the raw files (local only); the analysis
    reruns from that table.
- **The tonic level as a third observation** (Parts 15–16, exploratory variants): both failed
  validation. The balanced floor (`tonic14b`) fixes the overconfidence, but some drift still
  leaks in. Neither changes a verdict. The differenced tonic (`tonic14d`, Part 17) handles the
  drift but narrowly fails on scale, so it was not run on the 50.
  - **The calibrated version (`tonic14c`, Part 18) passes validation on fresh seeds.** On the 50
    every verdict matches the main analysis, and the estimates agree to within 0.03.
- **Open:** the rest of the plausibility check (flat stretches, sizes, shapes), a comparison with
  other methods.

## Stage 2: the input

```bash
PYTHONPATH=src python -m eda.run real-pulses
```

Reads only the final run's output, which is in the repository (no MATLAB, cluster or raw data):

| File | One row per | Columns |
|---|---|---|
| `data/real50_pulses.csv` | pulse | `id`, `group` (film group), `time_s` (recording clock), `u` (impulse), `height_uS` (the response it makes), `marker` (block code: −1 baseline, 1xx neutral, 2xx negative, 3xx positive film, empty between), `block` (baseline / film / other), `valence`, `window` |
| `data/real50_windows.csv` | 200 s window | `id`, `group`, `tau_rise`, `tau_decay` (s), `lambda1`, `lambda2`, `r2`, `selection`, … |

**Threshold (fixed 2026-11-14, before H1–H6 were looked at):** the main analysis uses pulses with
`height_uS` ≥ 0.05 µS (test F1 0.96, recall 0.98, precision 0.96); all returned pulses, ≥ 0.02 µS
(F1 0.82), are the secondary check. The full per-sample fit of each
participant (input, reconstruction, tonic, phasic, block code) is in
`hpc/own/final/output/own4_p1000_k14_f3_s8/P<id>_own4_signals.csv`. The affect slider (H6) is in
the raw recordings (`eda.dataset.load`).

## Layout

```
docs/EXPERIMENTS.md          the story, every number's source, current status, next steps
docs/dataset_check/          the POPANE metadata workbook the quality screen reads
src/stage2/                  STAGE 2: PULSES -> AROUSAL STATE (MATLAB; docs/EXPERIMENTS.md Part 13)
  s2_filter_mpp.m, s2_filter_bin.m   the Data Challenge 2 filters, arithmetic unchanged
  s2_test_dc.m               reproduces the Data Challenge's saved results (needs its folder)
  s2_observations.m          one participant's 1 s bins: pulse, log height, block code
  s2_synth.m                 validation on 400 synthetic sessions -> data/stage2_synth_scores.csv
  s2_run.m                   the 50 -> data/stage2_params.csv, data/stage2_states_00{5,2}.csv
  s2_plot.m                  one participant's figure -> results/figures/stage2_state_P<id>.png
  s2_plot_all.m              all 50, one page each -> results/figures/stage2_states_real50.pdf
  s2_tests.m                 H1-H6 as set in Part 14 -> data/stage2_tests.csv,
                             data/stage2_block_summary.csv, results/figures/stage2_hypotheses.png
  s2_filter_tonic.m          variants tonic14 (Part 15) and tonic14b (Part 16, opt.floor: the
                             balanced floor), both failed validation: + tonic level every 14 s
  s2_tonic_input.m           each tonic variant's input and options (tonic, tonic14b/d/c)
  s2_tonic_calibrate.m       tonic14c (Part 18): the tonic link from the pulse-only state
  s2_read_raw.m              reads columns of a raw POPANE file (data/raw/Study6, local only)
  s2_slider.m                the slider on the stage-2 bins -> data/stage2_slider_1hz.csv
  s2_slider_eval.m           state vs slider (Part 19) -> data/stage2_slider_eval[_tonic14c].csv,
                             results/figures/stage2_slider_eval[_tonic14c].png
  s2_synth_tonic.m, s2_run_tonic.m   their synthetic check and real run -> data/*_tonic.csv,
                             data/*_tonic14b.csv (second argument 'tonic14b');
                             s2_tests(pwd, 'variant', 'tonic' | 'tonic14b') -> data/stage2_tests_<v>.csv
src/own/                     OUR DECONVOLUTION (MATLAB)
  own_config.m               every setting, with where it comes from (paper, DC code, ours)
  own_run.m                  one session: 200 s windows every 100 s, random starts, selection
  own_deconvolve.m           one window: phase 1 FOCUSS+, phase 2 GCV-FOCUSS+, tau and tonic steps
  own_*.m                    the pieces (matrices, tonic basis, FOCUSS+, GCV, shape and tonic steps)
  own_test.m                 tests on made-up data (run after any change)
  make_own_bundle.sh         tuning bundle for the cluster; its comments hold every round's settings
  make_own_final_bundle.sh   the final run (real P + test V); make_own_resynth_bundle.sh: R
  run_own.sbatch             the cluster job array (NYU Torch)
src/eda/                     PYTHON: data, the solver's input, synthetic sessions, scoring
  run.py                     command-line entry point: every command, in run order
  config.py, dataset.py      paths and shared numbers; reading the POPANE files
  quality.py                 first screen: 187 -> 119 usable participants
  cohort_profile.py          model-free profile of every usable participant
  selection.py               the 50 analysed: baseline-only rule, then a seeded draw per group
  preprocess.py              the solver's input: every step in one order (prepare), its checks
  noise.py                   noise matched to roughness, slow noise, spectra
  tbme_synth.py, synth_v1.py the first generator and its runs (tuning rounds 1-5, first final run)
  generator.py               the frozen generator from round 6 on (tuning U, test V)
  tuning.py                  scoring every tuning round, and its diagnostics
  runs.py                    final run, re-synthesis, their figures, the stage-2 pulse table
  scr_shapes.py              model-free SCR shapes on the waveforms (a description only)
  tbme_export.py, tbme_score.py   the solver's input format; scoring against the truth
  figures.py                 every figure
  own_preprocess.py          compatibility only: the old single module, re-exporting the above
data/                        every result table (csv) and the synthetic truth (tbme_synth/*.npz)
results/figures/             preprocessing figures, generator inspection, deconvolution PDFs
hpc/own/                     cluster output: tune/ (rounds), final/, resynth/, tune_pre/ (unscorable)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Commands run from the project root as `PYTHONPATH=src python -m eda.run <command>`. Running it
with no command prints them all.

## Reproducing

**From the repository alone** (no raw data, no cluster):

| Command | Recomputes |
|---|---|
| `real-pulses` | the stage-2 tables |
| `test50-score` | the reported accuracy on the 50 test sessions (`data/own_test50_scores.csv`) |
| `resynth-score` | the re-synthesis score (`data/own_resynth50_scores.csv`) |
| `final-plots`, `resynth-plots` | the deconvolution figures |

`own-score` and `final-score` also run from the repository. For tuning rounds 5–7 and the first
final run, the repository keeps pulses, windows, logs and the score tables, but not the large
per-sample signals. So their pulse scores are recomputed, and their tonic measures
(`phasic_taken` and the like) come out blank. The signals come back by re-running that run's
bundle.

**From the raw data.** POPANE Study 6 (Behnke et al., *Scientific Data* 9:10, 2022), CC-BY, on
OSF, DOI 10.17605/OSF.IO/94BPX, 19 GB, not in this repository. Unzip `Study6.zip` to
`data/raw/Study6/`, so that files look like `data/raw/Study6/S6_P99_All.csv`. Then run the commands
in the order `run.py` lists them. The current pipeline is:

```
quality -> profile -> select-measures -> select -> final-export (writes the real inputs P*_own4)
-> synth-check -> tune50-export -> [cluster: make_own_bundle.sh, own-score]
-> test50-export -> [cluster: make_own_final_bundle.sh] -> test50-score, real-pulses
-> resynth-export -> [cluster: make_own_resynth_bundle.sh] -> resynth-score
```

The first generator's steps (`synth`, `own-check`, `own4-export`, `noise-spectrum`) are needed
only to reproduce rounds 1–5. `final-export` still reads two of their tables, which are in `data/`.

**On the cluster** (MATLAB R2025b, NYU Torch): each `make_own_*bundle.sh` writes a zip of the code,
inputs and a `jobs.txt` (one line per session and setting). Upload it to a new folder, unzip it,
then run `sbatch --mem=8GB run_own.sbatch`. Pack the output with `tar -czf` (zip is not
installed), download it, extract it into `hpc/own/<run>/`, then score it. The step-by-step version
is in docs/EXPERIMENTS.md, How to run a round. The MATLAB tests: `addpath src/own; own_test`.
