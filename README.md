# Decoding sympathetic arousal from skin conductance — branch `own`

Course project for BE-GY 9603 / ECE-GY 9153 A, Neural and Physiological Signal Processing.
Komvopoulos Eleftherios (ek4538) and Piyush More (pm4081). Report, slides and code are due
November 25, 2026.

## Two branches

| Branch | What it holds |
|---|---|
| `black-box` | Everything up to and including the published Amin & Faghih (2021) solver, run unmodified (encrypted MATLAB). Its own README and experiment log tell that story. Frozen. |
| `own` (this one) | **Our own implementation** of the same algorithm, with full control of every setting, tuned on synthetic sessions with a known truth. Independent: it runs without anything from `black-box`. |

We moved to our own implementation because the black box's settings could not be changed: its
response shape (τ) always stayed at its built-in prior, whatever the data said. We did not
invent a new algorithm. We wrote the published one ourselves, then tuned its settings on our data.

## Where we are

**Read [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) first.** It explains what we built, the four
tuning rounds, and every decision.

- **Best setting so far:** `own4_p1000_k14_f3`. On the 18 tuning sessions its median F1 is 0.84,
  with precision 0.90, timing within 0.17 s, and τ typically within 6% (rise) and 10% (decay) of
  the truth. The black box never moved τ off its prior.
- **Not done yet:** one more tuning round (tonic knot spacing), then the 18 test sessions, then
  real participants, stage 2 (arousal state), hypotheses H1–H6, report and slides.

## Layout

```
docs/
  EXPERIMENTS.md         the story, the verdicts and the current status
  dataset_check/         first look at POPANE, and the metadata the quality screen reads
src/own/                 OUR DECONVOLUTION (MATLAB)
  own_config.m           every setting, with where it comes from (paper, DC code, ours)
  own_run.m              one session: 200 s windows, random starts, selection, joined output
  own_deconvolve.m       one window: phase 1 FOCUSS+, phase 2 GCV-FOCUSS+, tau and tonic steps
  own_*.m                the pieces (matrices, tonic basis, FOCUSS+, GCV, shape and tonic steps)
  own_test.m             tests on made-up data (run this after any change)
  own_diagnose.m         made-up windows under several settings
  make_own_bundle.sh     packs inputs + code for the cluster; holds each round's settings
  run_own.sbatch         cluster job array (Torch)
src/eda/                 Python: data, synthetic sessions, the solver's input, scoring
  own_preprocess.py      the solver's input, every step in one order; the tuning scores
  config.py              paths and shared numbers, with the reason for each
  dataset.py             reads the POPANE csv files and the metadata workbook
  quality.py             automatic screen: 187 -> 119 usable participants
  cohort_profile.py      model-free description of each participant
  tbme_synth.py          36 synthetic sessions with known impulses, drawn from real profiles
  tbme_export.py         input files in the solver's format (named after the TBME paper's format)
  tbme_score.py          scores the solver's output against the truth
  figures.py             plots
  run.py                 command-line entry point, lists every command
data/                    result tables, plus the synthetic truth in tbme_synth/
results/figures/         preprocessing figures
hpc/own/tune/            cluster output (csv) and logs of every tuning round
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Raw data (19 GB) is not in this repository.** POPANE Study 6 (Behnke et al., *Scientific Data*
9:10, 2022), CC-BY, on OSF under DOI 10.17605/OSF.IO/94BPX. Unzip `Study6.zip` to
`data/raw/Study6/`, so that files look like `data/raw/Study6/S6_P99_All.csv`.

You don't need it to read the results: `PYTHONPATH=src python -m eda.run own-score` reads only
files in this repository and rebuilds `data/own_tune_scores.csv`.

To rebuild everything from the raw data, run the commands in the order `run.py` lists them
(`quality`, `profile`, `synth`, `own-check`, `own4-export`), then the cluster steps in
docs/EXPERIMENTS.md.
