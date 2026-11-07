# Decoding sympathetic arousal from skin conductance during emotional film viewing

Course project for BE-GY 9603 / ECE-GY 9153 A, Neural and Physiological Signal Processing.
Komvopoulos Eleftherios (ek4538) and Piyush More (pm4081). Report, slides and code are due
November 25, 2026.

We take finger skin conductance, recover the sudomotor nerve impulses behind it by deconvolution,
and estimate a hidden sympathetic-arousal state from those impulses with a marked point process
filter. The question is whether that state separates emotional from neutral film viewing, and
whether it carries valence or only intensity.

## Where we are

**Read [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) first.** It tells the whole story: what we tried,
what each test showed, and what was decided.

- **Deconvolution:** we dropped our own method and cvxEDA, and now run the published
  Amin & Faghih (2021) solver unmodified on the NYU Torch cluster.
- **Validated on synthetic data:** with the paper's preprocessing, median F1 is 0.84 when a
  person's true response shape matches the solver's built-in prior, and 0.70 when it doesn't.
- **Main weakness:** the solver always returns its prior response shape, and real participants
  look like the mismatched case. Two attempts to measure the true shape per person failed their
  own checks, so event timing and rate are more trustworthy than response sizes.
- **Not done yet:** only 5 real participants have been run. The 119-participant run, stage 2 on
  those events, hypotheses H1–H6, the report and the slides are all still ahead.

## Layout

```
docs/
  EXPERIMENTS.md         the story, the verdicts and the current status
  proposal_methods.md    the proposal, including hypotheses H1-H6
  literature_review_notes.md
  dataset_check/         first look at POPANE, and the metadata the quality screen reads
src/eda/
  config.py              every path and tunable number, with the reason for each
  dataset.py             reads the POPANE csv files and the metadata workbook
  quality.py             automatic screen: 187 -> 119 usable participants
  cohort_profile.py      model-free description of each participant
  tbme_export.py         the paper's preprocessing; writes the solver's input files
  tbme_synth.py          36 synthetic sessions with known impulses, drawn from real profiles
  tbme_score.py          scores solver output against the truth; most of the tests live here
  tbme_real.py           summarises solver output on real participants
  tau_measure.py         tau from single isolated responses (passed synthetic, failed real)
  tau_shared.py          one tau per person (failed)
  preprocess_view.py     compares real, synthetic and the authors' own input
  state.py               stage 2: impulses -> arousal state (filter, smoother, EM)
  figures.py             plots
  run.py                 command-line entry point, lists every command
  archive/               superseded: our deconvolution and the cvxEDA comparison
src/tbme/
  tbme_run.m             runs the authors' solver on one session; their .p code is not modified
  run_tbme.sbatch        cluster job array
  make_bundle.sh         packs inputs + solver into a zip for the cluster
  tbme_smoke.m           one-off check: the authors' code on their own data
  probe/                 one-off check: does the solver filter its input?
data/                    every result table, plus the synthetic truth in tbme_synth/
results/                 every figure
hpc/                     cluster output (csv) and logs, for every run so far
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Two things are not in this repository:

1. **Raw data (19 GB).** POPANE Study 6 (Behnke et al., *Scientific Data* 9:10, 2022), CC-BY, on
   OSF under DOI 10.17605/OSF.IO/94BPX. Unzip `Study6.zip` to `data/raw/Study6/`, so that files
   look like `data/raw/Study6/S6_P99_All.csv`.
2. **The authors' solver.** Encrypted MATLAB code, theirs to distribute:
   `git clone https://github.com/computational-medicine-lab/Tonic-Phasic-Decomposition-TBME.git external/Tonic-Phasic-Decomposition-TBME`
   (we used commit d6e5eb1).

You don't need either to read the results. `score-tbme` and `paper-test` read only files in this
repository, so those two can be rerun without the raw data or the cluster.

## Running

Commands run from the project root. `src/eda/run.py` lists all of them.

```bash
export PYTHONPATH=src
python -m eda.run quality         # 187 -> 119 usable, needed by everything else
python -m eda.run profile         # describes each participant, writes the unfiltered inputs
python -m eda.run synth-tbme      # builds the 36 synthetic sessions
python -m eda.run paper-export    # paper's preprocessing on 5 real + 8 synthetic
src/tbme/make_bundle.sh P53_pp P115_pp S4_pp
```

Upload `hpc/tbme_bundle.zip` to the cluster, unzip it, and run `sbatch run_tbme.sbatch`. Each
session takes about 15 minutes, and the output appears in `output/`. Copy the `S*` runs to
`hpc/synth/output/` and the `P*` runs to `hpc/real/output_pp/`, then run
`python -m eda.run paper-test`.

`paper-export` covers only the 13 test sessions so far; see "Next steps" in the experiment log.

## Key references

- Amin & Faghih, *IEEE TBME* 68(5):1726, 2021, and its supplement — the method we follow
- Behnke et al., *Scientific Data* 9:10, 2022 — the POPANE dataset
- Greco et al., *IEEE TBME* 63:797, 2016 — cvxEDA
- Wickramasuriya & Faghih, *IEEE Access* 8:68499, 2020 — the marked point process filter
- Boucsein et al., *Psychophysiology* 49:1017, 2012 — publication recommendations for EDA
