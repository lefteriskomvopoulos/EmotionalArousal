# Working in this repository (people and coding agents)

Read [README.md](README.md) for what exists and how to run it, and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for why things are the way they are. This file holds
the rules that keep the results valid.

## What is frozen

Stage 1 (skin conductance → impulses) is finished. These are fixed, and the reported numbers
depend on them:

| Frozen | Where |
|---|---|
| The deconvolution setting `own4_p1000_k14_f3_s8` | `src/own/make_own_final_bundle.sh` (args), `src/own/own_config.m` (defaults) |
| The input chain (1 Hz low-pass, 4 Hz, suffix `own4`) | `src/eda/preprocess.py` (`prepare`, `VARIANTS`) |
| The synthetic generator and its seeds | `src/eda/generator.py` |
| The 50 participants and the selection rule | `src/eda/selection.py`, `data/selection.csv` |
| The test sessions V and their truth | `data/tbme_synth/V*_own4.npz`, never used for tuning |

To try something different, add a new setting or command with a new name and keep the old one.
Do not edit a frozen one in place. A new deconvolution setting means a new tag, tuned on the 18
tuning sessions (U) only. It is tested once on V, and recorded as a round (below).

## Never overwrite a kept run's inputs or truth

These commands rewrite the solver inputs (`data/tbme_input/`) and the truth (`data/tbme_synth/`)
that existing cluster outputs were produced from and are scored against:

| Command | Rewrites |
|---|---|
| `synth`, `own-export`, `own4-export` | S sessions |
| `final-export` | real inputs P, and F, T |
| `tune50-export` | U |
| `test50-export` | V |
| `resynth-export` | R |

They are deterministic (fixed seeds), but a code change in between makes the new truth disagree
with the old output, and the run can no longer be scored. One preliminary run was lost exactly
this way. Before running one, back up the files it writes, then compare after. Scoring commands
only rewrite their score table. Back that up too if you want to compare.

## Method rules

These were each learned the hard way (docs/EXPERIMENTS.md):

1. **Judge on known truth, never on fit alone.** A near-perfect fit can hide a wrong response
   shape.
2. **Tune on U, test once on V, then run on real data.** Never choose anything by looking at V or
   at the real participants' results.
3. **No cohort calibration factors in the generator.** Each session copies its own participant;
   factors fitted across participants ran away and overfit.
4. **No priors from the solver's own outputs.** That is circular. The paper's prior comes from a
   different dataset; round 7 tested no prior against it (no better, less stable).
5. **Set the pass criterion before the run, and do not keep refining a measurement or generator
   until it passes.** The waveform-shape measurement (`scr_shapes.py`) failed its check and is kept
   as a description only, for this reason.
6. **Selection uses the baseline only.** Film-period responding is the outcome of H1–H3.
7. **Stage-2 checks of the pulses must be blind to the hypotheses**: plausibility, not whether
   H1–H6 come out as hoped.
8. **One change per tuning round.** Record its settings in the comments of
   `src/own/make_own_bundle.sh`, and its question, result and decision in docs/EXPERIMENTS.md.

## Conventions

- **Session names:** `<prefix><id>_<input>`, for example `V53_own4`.
  - P: real participant
  - S: first generator (rounds 1–5)
  - F, T: first final run's tests
  - U: tuning from round 6 on
  - V: final test
  - R: re-synthesis
  - A trailing `s` (`_own4s`) is the slow-noise variant, unused from round 6 on.
- **Setting tags:** `own4_p1000_k14_f3_s8` reads as:
  - input `own4`;
  - prior weight 0.1/1000 (doubled at 4 Hz);
  - tonic knots every 14 s;
  - phase-1 sparsity limit 10⁻³ (doubled);
  - 8 random starts.
  
  Suffixes are variants. Outputs live in `hpc/own/<run>/output/<tag>/`.
- **Units are µS throughout.** The solver runs in µS (normalising hurt, round 6). Pulses below
  0.02 µS are not returned.
- **Entry point:** `PYTHONPATH=src python -m eda.run <command>`. Every command is listed with its
  inputs, outputs and running time in `src/eda/run.py`. Add new commands there, in run order.
- **Tables:** results go to `data/*.csv` and figures to `results/figures/`. Both are named in the
  command's docstring.
- **MATLAB:** after any change under `src/own/`, run `addpath src/own; own_test`.
- **Cluster (NYU Torch):**
  - one new folder per run;
  - `sbatch --mem=8GB run_own.sbatch`, which keeps the array under the per-user memory limit;
  - `tar -czf` to pack the output (zip is not installed);
  - a job that dies starting MATLAB's parallel pool (exit 9, out of memory) is resubmitted alone
    with `--array=<k> --mem=16GB`.

## Starting stage 2

`real-pulses` writes `data/real50_pulses.csv` and `data/real50_windows.csv` from the committed
final-run output (README, "Stage 2: the input"). The film group is in `data/selection.csv`; the
block of each pulse is in its `marker`. The affect slider (H6) comes from the raw recordings
(`eda.dataset.load`, column `affect`). The first stage-2 code (`state.py`) is on branch
`black-box` and has not been brought over yet. Known stage-1 limits that bear on stage 2:
- decays run long on fast and typical shapes and short on slow ones;
- false pulses gather where the tonic rises fast;
- about 36% of real isolated responses recover as fast as they rise, a shape the model cannot
  produce.

All three are in docs/EXPERIMENTS.md, Parts 9–11.
