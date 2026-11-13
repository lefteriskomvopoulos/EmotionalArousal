# Decoding sympathetic arousal from skin conductance — branch `own`

Course project for BE-GY 9603 / ECE-GY 9153 A, Neural and Physiological Signal Processing.
Komvopoulos Eleftherios (ek4538) and Piyush More (pm4081).

The pipeline: skin conductance of POPANE Study 6 → **sudomotor impulses** (stage 1, our own
implementation of Amin & Faghih, IEEE TBME 2021) → **arousal state** (stage 2) → hypotheses H1–H6.
Stage 1 is done and frozen on this branch. Stage 2 starts from one table (below).

## Start here

1. This README: what exists, where, and how to run it.
2. **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**: the whole story in order. Every experiment, its
   result, and the decision it justified, with the file each number comes from.
3. [AGENTS.md](AGENTS.md): the rules for working in this repository (for people and coding agents).

## Two branches

| Branch | What it holds |
|---|---|
| `black-box` | Everything up to and including the published Amin & Faghih (2021) solver, run unmodified (encrypted MATLAB), and the first stage-2 code (`state.py`). Frozen. |
| `own` (this one) | **Our own implementation** of the same algorithm, with every setting under our control, tuned on synthetic sessions with a known truth, then run on the real participants. Runs without anything from `black-box`. |

We wrote our own implementation because the black box's settings could not be changed: its
response shape always stayed at its built-in prior. The algorithm is the published one; its
settings were tuned on our data.

## Where things stand

- **Frozen setting `own4_p1000_k14_f3_s8`** (`src/own/make_own_final_bundle.sh`). On 50 unseen
  synthetic test sessions built from the 50 analysed participants: F1 0.82, recall 0.85,
  precision 0.85, timing within 0.16 s, size rank correlation 0.98.
- **50 real participants deconvolved** (20 neutral, 10 positive, 10 negative, 10 mixed film
  groups; `data/selection.csv`). Their pulses: `data/real50_pulses.csv` (`real-pulses`, below).
- **Known limits:** slow responses (half-recovery > 7 s) lose precision (0.74) and their decay
  comes out 37% short; on fast and typical shapes, which most real participants have, decays run
  long. The full list is in docs/EXPERIMENTS.md, Current status.
- **Open:** round 7 (no prior against the frozen setting), re-synthesis session R115, a check that
  the real pulses make sense before stage 2, a comparison with other methods, stage 2, H1–H6.

## Stage 2: the input

```bash
PYTHONPATH=src python -m eda.run real-pulses
```

Reads only the final run's output, which is in the repository (no MATLAB, cluster or raw data):

| File | One row per | Columns |
|---|---|---|
| `data/real50_pulses.csv` | pulse | `id`, `group` (film group), `time_s` (recording clock), `u` (impulse), `height_uS` (the response it makes), `marker` (block code: −1 baseline, 1xx neutral, 2xx negative, 3xx positive film, empty between), `block` (baseline / film / other), `valence`, `window` |
| `data/real50_windows.csv` | 200 s window | `id`, `group`, `tau_rise`, `tau_decay` (s), `lambda1`, `lambda2`, `r2`, `selection`, … |

The solver keeps only pulses of at least 0.02 µS. The planned robustness check repeats the
hypotheses with pulses of at least 0.05 µS (`height_uS`). The full per-sample fit of each
participant (input, reconstruction, tonic, phasic, block code) is in
`hpc/own/final/output/own4_p1000_k14_f3_s8/P<id>_own4_signals.csv`. The affect slider (H6) is in
the raw recordings (`eda.dataset.load`).

## Layout

```
docs/EXPERIMENTS.md          the story, every number's source, current status, next steps
docs/dataset_check/          first look at POPANE, and the metadata the quality screen reads
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
