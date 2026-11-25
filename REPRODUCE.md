# Reproducing the results

One command recomputes the project's results and compares every recomputed file with the one we
shipped:

```bash
./reproduce.sh
```

It works on a scratch copy, so the delivered folder is never changed. It prints one line per step,
then a comparison, and ends with `RESULT: every recomputed result matches the shipped one`, or
with a list of the problems. The full table is written to `reproduce_report.csv`.

## Which 50 participants

All 97 participants who passed selection were deconvolved with the same frozen setting: the 50
drawn at random in `hpc/own/final/`, and the other 47 in `hpc/own/extra/`. The report analyses the
drawn 50 with six of them replaced by participants of the same film group from the other 47,
because the stage-2 state filter that uses pulse sizes did not converge for those six (report,
Section II-B; the swap is recorded in `src/eda/new50.py`). The check recomputes the drawn 50's
tables, the other 47's pulse tables, and the analysed 50's real-data numbers
(`data/new50_numbers.csv`, the drawn 50 next to the analysed 50). The shipped stage-2 results
(`data/stage2_*`) are those of the drawn 50.

## What is in this folder

| Folder | What it holds |
|---|---|
| `src/eda/` | Python, stage 1: reading POPANE, the solver's input, synthetic sessions with known truth, scoring, figures. `PYTHONPATH=src python -m eda.run` lists every command. |
| `src/own/` | MATLAB, stage 1: our implementation of the Amin & Faghih (2021) deconvolution. |
| `src/stage2/` | MATLAB, stage 2: the arousal state estimated from the pulses, the H1–H6 tests, the slider. |
| `hpc/own/*/output/` | Every deconvolution run on the cluster: pulses, windows and the per-sample fit (`final`: the 50 drawn and their test sessions; `extra`: the other 47 who passed selection; `resynth`; `tune`). |
| `data/` | Result tables, the test sessions' truth (`tbme_synth/`) and the solver's inputs (`tbme_input/`). |
| `results/figures/` | The figures. |
| `docs/EXPERIMENTS.md` | Every experiment in order: its question, its result, the decision it led to, and the file each number comes from. |
| `archive/paper_solver/` | The frozen record of the published solver, run unmodified, and its comparison with cvxEDA. It is not rerun here (see below). |

`README.md` gives an overview of the project. The report and slides are submitted separately, and
the cluster's log files are left out.

## Requirements

- Python 3.9 or newer, with the packages in `requirements.txt`:
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```
  `reproduce.sh` uses `.venv/bin/python` if it exists, otherwise `python3`.
- MATLAB, with the Optimization Toolbox and the Statistics and Machine Learning Toolbox. The
  deconvolution ran on R2025b (NYU's cluster), and the shipped stage-2 results were made with
  R2025a. We checked the whole folder on R2026b (below). The Parallel Computing Toolbox is optional
  and only speeds up `--solver`. Without MATLAB, the Python part still runs and the stage-2 steps
  are reported as skipped.
- `bash` and `rsync` (both standard on macOS and Linux).

## Options

| Command | What it adds |
|---|---|
| `./reproduce.sh --fast` | Skips stage 2's synthetic validations, which are the slowest part. |
| `./reproduce.sh --rebuild` | First rebuilds every solver input, the participant selection and the test sessions' truth from the raw recordings. Needs the raw data (below). The recorded step times add up to about 80 minutes; several steps have no recorded time. **Not yet tested end to end.** |
| `./reproduce.sh --solver V53_own4` | Also reruns the deconvolution on one session (P, V or R, any id) with the frozen setting, and compares its pulses, windows and fit with the shipped run. About an hour per session. **Not yet tested end to end.** |

The plain `./reproduce.sh` was tested from a fresh copy of this folder, both with and without the
raw data (MATLAB R2026b, Python 3.12), and took 23 to 30 minutes on a laptop. `--fast` runs the
same steps without the synthetic validations; by the same steps' times, that takes about 10 minutes.
| `--work DIR`, `--raw DIR` | Choose the scratch folder, or say where the raw data is. |

**Raw data.** POPANE Study 6 (Behnke et al., *Scientific Data* 9:10, 2022) is on OSF under
CC-BY, DOI 10.17605/OSF.IO/94BPX (19 GB). Unzip `Study6.zip` to `data/raw/Study6/`, so that the
files look like `data/raw/Study6/S6_P99_All.csv`. When these files are present, the plain
`./reproduce.sh` also runs the steps that read them: the H6 test and the slider table.

## How to read the comparison

| Status | Meaning |
|---|---|
| IDENTICAL | The same bytes. |
| MATCH | The same numbers to a relative 10⁻⁶, which is the last printed digits. |
| CLOSE | The same to 10⁻³ but not to 10⁻⁶, which is rounding on a different machine or MATLAB version. Listed, but not counted as a failure. |
| PARTIAL | Matches, except for cells left blank because that run's per-sample signals are not shipped (below). |
| VERSION-DEPENDENT | A stage-2 table that matches except in places another MATLAB version can change (next section). Listed with counts, but not counted as a failure. |
| REGENERATED | A figure: rewritten, but not compared pixel by pixel. |
| NEW | Written, but not shipped, so there is nothing to compare it with. |
| NO RECORD | A solver input with no per-sample record of the run shipped (the tuning sessions U). |
| FAIL, NOT REGENERATED, STEP FAILED | A problem. The log of each step is in `logs/` in the scratch folder. |
| SKIPPED | The step did not run (no MATLAB, no raw data, or `--fast`). |

The check also confirms that every shipped solver input (`data/tbme_input/`) is exactly the signal
the solver recorded it ran on. At the end it lists the shipped files that no step recomputes in
the mode used.

## Known differences between MATLAB versions

Stage 2 estimates the arousal state with two iterative filters (MPP and binary) and keeps the MPP
filter's state only when that filter converges; otherwise it keeps the binary one. An iterative
fit can take a slightly different path on another MATLAB version, so a few numbers can change
without changing any result. The comparison allows these, and only these:

- **How a fit got there:** iteration counts and stop reasons.
- **Results of a filter run that was not used:** for example, the MPP filter's numbers for a
  participant whose MPP filter did not converge, where the binary filter's state is the one kept.
  Which filter is kept is still checked strictly.
- **The secondary H6 row** (state against raw valence). It includes a participant only if their
  averaged slider values are not exactly equal, and that last digit can differ between versions.
- **Up to 1% of sessions in a synthetic validation table**, where a filter that converges slowly
  stops at a slightly different point.

When we reran the shipped R2025a results on R2026b, every participant's chosen state matched to
10⁻⁶, and so did every H1–H5 test and the main H6 test. The secondary H6 row went from 45 to 44
participants (estimate −0.04 to −0.05, p 0.60 to 0.55). This row is reported, but no conclusion
rests on it. Any other difference is a FAIL.

## What is not rerun, and why

- **The deconvolution of every session.** The 150 final sessions (50 real participants, 50 test
  sessions, 50 re-synthesised signals) and the tuning rounds ran on NYU's cluster. One session
  took 51 to 90 minutes on 4 workers (real participants; the window times are in each run's
  `_windows.csv`). Their outputs ship in `hpc/own/*/output/`, and every score,
  table and stage-2 result is recomputed from them. `--solver` reruns any one session from its
  shipped input.
- **The tonic measures of tuning rounds 5–7 and of the first final run.** Their per-sample
  signals were too large to keep, so those cells come out blank (PARTIAL). Their pulse scores are
  recomputed.
- **H6 without the raw data.** The H6 test reads the slider from the raw recordings. Without them,
  those rows are marked "not tested" and left out of the comparison. The slider evaluation itself
  runs from the shipped 1 Hz table, `data/stage2_slider_1hz.csv`.
- **Earlier exploratory tables** from steps that were later replaced. They are kept as the record
  that `docs/EXPERIMENTS.md` cites, and they are counted in the final "not recomputed" list.
- **`archive/paper_solver/`.** The published solver is encrypted MATLAB, and it ran on the
  cluster. Its outputs and the cvxEDA comparison are kept there as a record.
- **`src/stage2/s2_test_dc.m`** checks the filters against the Data Challenge's own saved results,
  which are not part of this project.
