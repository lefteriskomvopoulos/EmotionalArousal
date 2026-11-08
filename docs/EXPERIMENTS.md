# Experiment log — branch `own`

What we built, how we tested it, what each tuning round showed, and what we decided. Read it top
to bottom. Every number comes from a saved table: `data/own_tune_scores.csv` (one row per setting
and session) unless another file is named.

## The short version

1. The published solver (branch `black-box`) could not be tuned: its τ always stayed at its
   built-in prior. So we **implemented the same algorithm ourselves** (Amin & Faghih, IEEE TBME
   2021), and checked it against the black box: same model, same output.
2. We rebuilt the **preprocessing** as one chain in one order, following the papers, and made the
   synthetic sessions as noisy as the real recordings.
3. We **tuned on 18 synthetic sessions** with known pulses and τ, four rounds so far. Median F1
   went from 0.47–0.61 (paper-like settings) to **0.84**, and τ now follows the truth.
4. The 18 **test sessions** are still unseen. They come once the settings are frozen, then the
   real participants.

## Terms

| Term | Meaning |
|---|---|
| τ rise, τ decay | How fast a skin conductance response rises and falls: the response *shape* |
| Prior | A penalty λ/(2σ²)(τ − μ)² that pulls τ toward the paper's μ = 0.65 s (rise), 2.77 s (decay), σ = 0.21 / 0.52 s |
| Tonic / phasic | The slow baseline (a cubic B-spline) / the fast responses on top of it |
| u | The impulses (sudomotor nerve bursts) that drive the responses |
| FOCUSS+, GCV-FOCUSS+ | The sparse solvers for u. Phase 1 keeps at most (peaks + 20) impulses; phase 2 picks the sparsity weight λ2 by generalized cross-validation (GCV) |
| F1, recall, precision | Scores for finding known impulses, matched one to one within 1 s. Recall: share of true impulses found. Precision: share of returned impulses that are real |
| Absorption | `phasic_taken`: share of each true response that the estimated tonic wrongly takes in. Ideal 0 |
| τ groups | Synthetic sessions by true τ: `prior` (drawn from the prior), `moderate` (around 0.4/2.0 or 1.0/4.0 s), `far` (around 1.2/5.0 s), ±10% |

## Part 1: why our own implementation

On the `black-box` branch the authors' encrypted solver passed our synthetic test, but its τ never
left the prior (0.65 / 2.77 s) whatever the true τ was. When the truth was far from the prior, about
46% of each response leaked into the tonic. Its prior strength, timing and limits are inside the
encrypted code, so none of this could be changed. Every workaround (measuring τ per person, a shared
τ, a finer rate) failed its checks.

So we wrote the algorithm ourselves. The rule: **keep the algorithm, tune its settings on our
data.** No new method is invented, and the paper's numbers are starting points, not targets.

## Part 2: what we implemented (`src/own/`)

The model of Amin & Faghih (2021): each response obeys τr·τd·y″ + (τr + τd)·y′ + y = u, so a window
of the signal is y = a·y0 + B(τ)·u + C·q + noise, with the tonic C·q a cubic B-spline. The cost:

  ½‖y − a·y0 − B u − C q‖² + λ1‖q‖² + λ2‖u‖ₚᵖ + prior on τr + prior on τd

under u ≥ 0, C q ≤ y, 0 ≤ y0 ≤ y1 and τ limits. It is solved by block coordinate descent
(supplement, section D) in each 200 s window, every 100 s:

- **Phase 1**, 30 times: u by FOCUSS+ from all ones, then τ and y0 (interior point), then q.
- **Phase 2**, until nothing moves: u by GCV-FOCUSS+ from the last u, then τ and y0, then λ1 by GCV
  and q.
- 4 random starts per window; the lowest-residual converged start whose τ rise is off its limits
  is kept.

`own_config.m` lists every setting and where it comes from (paper, Data Challenge code, or ours).
It also holds the Data Challenge algorithm (Faghih 2014: same solver, no tonic, no prior), not
tuned yet.

**Checks.**

| Check | Result |
|---|---|
| `own_test.m`, made-up data | All passed: DC gain 1, response matches the ODE, 13 of 13 pulses found, converged |
| Against the black box's own output (P53, paper preprocessing): its phasic part rebuilt from its saved τ and u with our matrices | R² 1.00000, gain 1.000, u shifted by +2 samples (an indexing difference, not physics). Its tonic rebuilt with our basis: R² 0.997–0.9999 |
| `own_diagnose.m`, made-up windows | A weaker prior moved a far window's τ to 1.18 / 4.79 s (true 1.20 / 5.00) |

Our implementation is the black box's model, but we now control every setting. (The script of the
second check needs the black box's output files, so it stays out of this branch.)

## Part 3: the solver's input (`src/eda/own_preprocess.py`)

All steps run at the recording's own rate, with a single down-sampling at the end. Amin & Faghih
(IEEE Access 2019): "We perform all the preprocessing in the original sampling frequency".

| Step | What |
|---|---|
| 1 raw | 1000 Hz recording |
| 2 gaps | linear interpolation (the steps below need a gap-free signal) |
| 3 jumps | peaks of prominence 0.1 µS on the 10 ms change; 0.5 s around each replaced by a cubic B-spline (2021 supplement), drawn through 0.1 s means (ours) |
| 4 low-pass | zero-delay FIR, Hamming, 6.4 s long; 0.5 Hz (2021 supplement) or 1 Hz (variant `own4`) |
| 5 cut | first baseline sample to last film sample, after the filter, so its edges fall outside |
| 6 down-sample | to 2 Hz (`own`) or 4 Hz (`own4`): the only rate change |
| 7 shift | a constant, only if the signal goes below 0 µS (an amplifier offset) |

`own-check` ran all 119 usable participants through it: no jump inside any kept segment, gaps
under 12 s, at least 42 s of recording before each segment (`data/input_check.csv`). The 0.5 Hz
filter keeps half the signal at 0.5 Hz and 0.1% at 1 Hz (`data/filter_response.csv`).

Figures: `results/figures/stages_<name>.png` (every step, 5 real + 5 tuning sessions) and
`raw_atlas_*.png` (all 119 participants and the 18 tuning sessions, unprocessed).

## Part 4: the synthetic sessions and the test scheme

36 sessions (`tbme_synth.py`), each copied from the measured profile of one real participant:
their tonic floor, SCR rate and sizes, noise, gaps and timeline, with impulses and τ we know. Sizes
are capped at the largest SCR counted on the source participant.

**Noise had to be matched.** White noise matched at 100 Hz is mostly removed by the low-pass, while
real signals keep slow disturbances, so the first synthetic inputs were 3.5× smoother than real
ones. `matched()` adds slow noise until each session is as rough as its source participant, after
the same chain: 0.0026 vs 0.0028 µS at 2 Hz, and 0.0017 vs 0.0016 µS at 4 Hz
(`data/input_check_own4.csv`). The true pulses do not change.

**Split, drawn once** (`data/own_split.csv`): 18 tuning + 18 test sessions, 6 + 6 per τ group.

| | prior | moderate | far |
|---|---|---|---|
| tune | S1 S4 S22 S25 S28 S34 | S2 S11 S17 S23 S26 S32 | S3 S6 S21 S27 S30 S36 |
| test | S7 S10 S13 S16 S19 S31 | S5 S8 S14 S20 S29 S35 | S9 S12 S15 S18 S24 S33 |

Settings are chosen on the tuning sessions only. Once frozen, they run once on the test sessions,
and only then on real participants. Tuning is judged on the known truth, never on fit alone: the
project showed repeatedly that a good fit can hide a wrong τ.

## Part 5: the tuning rounds

Each round changes one thing. Tags name the setting (`hpc/own/tune/output/<tag>/`):
`p1` / `p10` / `p100` / `p1000` = prior weight 0.1 (paper) / 0.01 / 0.001 / 0.0001; `k6`, `k14` =
tonic knots every 6 or 14 s; `f3`, `f4` = phase-1 λ limit 10⁻³, 10⁻⁴ (default 10⁻²); `own4` = the
1 Hz / 4 Hz input. The exact arguments of each round are in `src/own/make_own_bundle.sh` and its
comments. All numbers below are medians over the 18 tuning sessions, or over a group's 6.

### Round 1: prior strength × knot spacing (6 settings, 108 jobs)

| Setting | F1 | Recall | Precision | Far-group decay error |
|---|---|---|---|---|
| p1_k6 | 0.47 | 0.31 | 1.00 | −43% |
| p10_k6 | 0.49 | 0.32 | 1.00 | −37% |
| p100_k6 | 0.46 | 0.30 | 1.00 | −14% |
| p1_k14 | 0.61 | 0.44 | 0.98 | −39% |
| p10_k14 | 0.59 | 0.42 | 1.00 | −27% |
| p100_k14 | 0.58 | 0.41 | 1.00 | −9% |

- **Knots every 14 s beat 6 s** (F1 about 0.60 vs 0.47): with 6 s knots the tonic swallows the slow
  far-group responses (absorption 0.28–0.47 vs 0.13–0.25). Kept 14 s.
- **A weaker prior frees τ** in the far group, but it overestimated decay in the prior group
  (+37% at p100).
- **The real problem: too few pulses.** Precision was about 1.00 but only about half of the pulses
  ≥ 0.02 µS were found. Two causes in the code. In FOCUSS+ a zero never comes back, so phase 1 sets
  the upper limit for phase 2, and phase 1's λ limit (10⁻², our own guess: the paper gives none) was
  100× the phase-2 limit. And GCV sat at its upper limit in about 90% of windows.

### Round 2: phase-1 λ limit (4 settings, 72 jobs)

| Setting | F1 | Recall | Precision |
|---|---|---|---|
| p100_k14 (round 1, 10⁻²) | 0.58 | 0.41 | 1.00 |
| p100_k14_f3 (10⁻³) | **0.81** | 0.80 | 0.86 |
| p100_k14_f4 (10⁻⁴) | 0.77 | 0.84 | 0.74 |
| p10_k14_f3 | 0.79 | 0.81 | 0.77 |
| p10_k14_f4 | 0.71 | 0.86 | 0.64 |

- **10⁻³ fixed the missed pulses.** 10⁻⁴ starts finding false ones. p100 beats p10 on precision and
  on far-group decay (−22% vs −36%).
- **τ was still wrong in a telling way.** τ rise had a floor near 0.6 s: true rises of 0.3–0.4 s
  came out at 0.6–1.0 s. And decay was pulled toward the middle from both sides. Likely cause: the
  0.5 Hz filter and 2 Hz sampling smear each response by about a second, and the fit reads the
  smear as a slower response.

### Round 3: the input, 1 Hz low-pass at 4 Hz (1 setting, 18 jobs)

Same algorithm on the `own4` input. Impulses stay at 4 Hz, so u means the same as before. Twice the
samples doubles the data term, so every penalty weight is doubled to keep its balance with the fit
(prior, both sparsity limits, tonic limit).

| | p100_k14_f3 (2 Hz) | own4_p100_k14_f3 (4 Hz) |
|---|---|---|
| F1 | 0.81 | **0.85** |
| Recall | 0.80 | 0.86 |
| Precision | 0.86 | 0.92 |
| Timing error | 0.25 s | 0.17 s |
| Amplitude rank correlation | 0.93 | 0.96 |
| Rise error: far / moderate / prior | +8 / +41 / +35% | −2 / +19 / +10% |
| Decay error: far / moderate / prior | −22 / −4 / +15% | −23 / +3 / +14% |

- **The 4 Hz input wins on every detection measure.** F1 rose in 16 of 18 sessions, and precision
  rose rather than fell.
- **The rise floor dropped** to about 0.45–0.55 s. Only true rises of 0.3–0.4 s are still
  overestimated, near the resolution of a 4 Hz signal.
- **Decay did not change**, so the filter was not its cause. Both errors pointed at 2.77 s, the
  prior's mean.
- A session takes 7–39 min on the cluster (2 Hz: up to 15 min).

### Round 4: prior weaker or off (2 settings, 36 jobs)

| Setting (4 Hz) | F1 | Rise error | Decay error | Worst decay | Far decay | Prior-group decay |
|---|---|---|---|---|---|---|
| own4_p100_k14_f3 (round 3) | 0.85 | 8% | 14% | 29% | −23% | +14% |
| **own4_p1000_k14_f3** | 0.84 | **6%** | **10%** | 34% | **−4%** | +20% |
| own4_p0_k14_f3 (no prior) | 0.83 | 8% | 16% | 81% | +6% | +24% |

Rise and decay error: median of the absolute error over the 18 sessions.

- **p1000 is the best so far.** Far-group decay error went from −23% to −4%, and the rise bias is
  gone, for almost no F1.
- **No prior at all is unstable:** one far session's decay went to 9.5 s (true 5.2), and in the far
  group a median of only 64% of windows met the selection rule. The prior is needed, just much
  weaker than in the paper.
- **What's left:** without the prior's pull, decays of about 2–4 s come out 10–35% too long (S28
  3.71 vs 3.04 s, S11 4.23 vs 3.70 s). Two dense sessions with short decays (S26, S32) are right.
  So this is a separate upward bias, which the prior was partly hiding.

## Current status

**Best setting: `own4_p1000_k14_f3`.** 4 Hz input with a 1 Hz low-pass; prior weight 2·10⁻⁴;
knots every 14 s; phase-1 λ limit 2·10⁻³; GCV limits 2·10⁻⁴ (λ2) and 2·10⁻⁴ (λ1); one impulse per
signal sample (4 Hz); everything else as in `own_config.m`. By τ group, F1 is 0.85 (prior), 0.87
(moderate) and 0.80 (far). On the `black-box` branch the published solver reached 0.65–0.76 by
group (0.70–0.84 with the paper's preprocessing), on smoother synthetic inputs, so the numbers are
not like-for-like.

Open problems:

1. Decays of about 2–4 s come out 10–35% too long (round 4).
2. Very dense sessions stay poor: S36 (25 pulses a minute, many within 1 s of each other) has F1
   about 0.5 in every setting.
3. True rises under about 0.45 s are still overestimated, near what a 4 Hz signal can resolve.
4. The Data Challenge algorithm (Faghih 2014) is implemented but not tuned. It needs a tonic-free
   input; in the 2019 paper the tonic is removed after the low-pass, before resampling.

## Next steps

1. **Round 5: knot spacing** at the best setting, 10 and 20 s (14 s exists). Knot 14 was chosen in
   round 1, when recall was 0.4 and τ was stuck at the prior. A too-stiff tonic leaves slow drift
   that a longer decay absorbs, which could explain open problem 1.
2. Freeze the settings and run them **once on the 18 test sessions**.
3. **Real participants**: the 5 exported (`P*_own4`), then all 119.
4. Stage 2 (impulses → arousal state) and hypotheses H1–H6. That code is on `black-box`
   (`state.py`) and comes over when we get there.
5. Report and slides.

## How to run a round

1. Local: export the inputs (`own4-export`), set the round's settings in
   `src/own/make_own_bundle.sh`, run it. It writes `hpc/own_bundle.zip`.
2. Upload the zip to `~/Project/own/` on Torch. There: `unzip -o own_bundle.zip`,
   `cd own_bundle`, `sbatch run_own.sbatch`. One job per session and setting.
3. Download `output/` into `hpc/own/tune/output/`, then
   `PYTHONPATH=src python -m eda.run own-score`. It scores every setting found there and saves
   `data/own_tune_scores.csv`.

Before any code change: `own_test` in MATLAB (`addpath src/own; own_test`).

## Where everything is

| What | Where |
|---|---|
| Scores of every tuning setting, per session | `data/own_tune_scores.csv` |
| Cluster output and logs, every round | `hpc/own/tune/output/<tag>/`, `hpc/own/tune/slurm-own-*.out` |
| Settings of each round | `src/own/make_own_bundle.sh` (comments), `src/own/own_config.m` |
| Tuning / test split | `data/own_split.csv` |
| Truth of each synthetic input | `data/tbme_synth/S<k>_own.npz`, `S<k>_own4.npz` |
| Synthetic cohort (source participant, τ, measures) | `data/tbme_synth_cohort.csv`, `data/tbme_synth_check.csv` |
| Input checks | `data/input_check.csv`, `data/input_check_own4.csv`, `data/filter_response.csv`, `data/preprocess_stages.csv`, `data/raw_overview.csv` |
| Participant screen and profiles | `data/quality_study6.csv`, `data/profile_study6.csv`, `data/profile_study6_peaks.csv` |
| Figures | `results/figures/stages_*.png`, `raw_atlas_*.png`, `filter_response.png` |
