# Experiment log — branch `own`

What we built, how we tested it, what each step showed, and what we decided, in the order it
happened. Each step answers a question the previous one raised. Read it top to bottom. Every
number comes from a saved table: `data/own_tune_scores.csv` (one row per setting and session)
unless another file is named.

## The short version

1. The published solver (branch `black-box`) could not be tuned: its τ always stayed at its
   built-in prior. So we **implemented the same algorithm ourselves** (Amin & Faghih, IEEE TBME
   2021), and checked it against the black box: same model, same output.
2. We rebuilt the **preprocessing** as one chain in one order, following the papers, and made the
   synthetic sessions as noisy as the real recordings.
3. **Rounds 1–5** tuned on 18 synthetic sessions with known pulses and τ (the first generator).
   Median F1 went from 0.47–0.61 (paper-like settings) to 0.84–0.90, and τ followed the truth.
4. **Participant selection** (Part 6): 50 of the 119 usable participants (20 neutral, 10 positive,
   10 negative, 10 mixed), by one model-free rule on the baseline only, fixed before any real
   participant was deconvolved.
5. **The first final run** (Part 7) showed the first generator's sessions were unlike the real
   inputs in a way that matters to the solver (signal scale). So we **rebuilt the synthetic data**
   from the 50 selected participants (Part 8): a per-participant plug-in with no cohort correction
   factors, frozen 2026-11-11.
6. **Round 6** on 18 new tuning sessions froze `own4_p1000_k14_f3_s8` (F1 0.835).
7. **The final run** (Part 9), once: 50 unseen test sessions (F1 0.819, recall 0.850, precision
   0.846; slow responses worst) and the 50 real participants.
8. **The prior on τ** (Part 11): the paper's prior comes from a different dataset. A prior refitted
   to our own estimates (circular) and a prior measured on the waveforms (failed its check) were
   both dropped. Round 7 tests no prior against the paper's prior, weak.

Still to come: round 7's result, the re-synthesis score (Part 10), a check that the real pulses
make sense before stage 2, a comparison with other methods, then stage 2 and H1–H6.

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
| τ groups | First-generator sessions (rounds 1–5) by true τ: `prior` (drawn from the prior), `moderate` (around 0.4/2.0 or 1.0/4.0 s), `far` (around 1.2/5.0 s), ±10% |
| Shape classes | New-generator sessions (round 6 on) by true half-recovery, the time from peak to half recovery: `fast` < 4 s, `typical` 4–7 s, `slow` > 7 s |
| Visible pulses, F1 visible | Only true impulses ≥ 0.02 µS count as targets, since nothing can detect smaller ones. F1 from that recall and the precision: the headline score from round 5 on |
| Session prefixes | S: first generator (rounds 1–5). F, T: first final run's test sessions (flat / slow noise). U: round-6 tuning. V: final test. P: real participant. R: re-synthesis |

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

## Part 3: the solver's input (`src/eda/preprocess.py`)

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

### Round 5: knot spacing 10 and 20 s (2 settings, 36 jobs)

Knots every 14 s were chosen in round 1, when recall was 0.4 and τ was stuck at the prior. A
too-stiff tonic leaves slow drift that a longer decay absorbs, which could explain round 4's
long decays, so the spacing was tested again at the best setting.

| Setting (4 Hz) | F1 visible | Recall visible | Precision | Rise error | Decay error | Decay error far / moderate / prior |
|---|---|---|---|---|---|---|
| own4_p1000_k10_f3 | 0.90 | 0.89 | 0.95 | 6% | 10% | +1 / +3 / +23% |
| **own4_p1000_k14_f3** (round 4) | 0.89 | 0.92 | 0.90 | 6% | 10% | −4 / +7 / +20% |
| own4_p1000_k20_f3 | 0.89 | 0.94 | 0.91 | 6% | 6% | −1 / +6 / +24% |

Rise and decay error: median absolute error. By group: median signed error.

- **Spacing makes no difference** beyond what 18 sessions can resolve: F1 within 0.01, and the
  prior group's decays stay 20–24% long at every spacing. So the stiffness of the tonic is not
  the cause of the long decays, and 14 s stayed.
- **Frozen: `own4_p1000_k14_f3`** for the first final run (Part 7). Limitation to report:
  decays of 2–3 s about 20% too long.

## Part 6: the participant selection (`src/eda/selection.py`)

The real-data analysis uses 50 of the 119 participants who passed the first screen, balanced for
the hypotheses: 20 neutral, 10 positive, 10 negative, 10 mixed (H3 compares positive and negative;
H4 needs the mixed group). A perfect solver does not exist, so the analysed set is made of signals
the method can read, chosen on stated grounds.

| Stage | Rule | Result |
|---|---|---|
| 1 | First screen (`quality.py`, fixed before the proposal): the dataset's EDA flag, electrode noise, missing data, near-flat recordings, non-responders | 187 → 119 |
| 2 | Median baseline SCR ≥ 0.05 µS. Below it most responses lie in 0.02–0.05 µS, where the synthetic test recovered 58% at 65% precision, against 98% / 96% above. 0.05 µS is the top of the literature's minimum-SCR range (Boucsein et al. 2012) | 119 → 97 (22 out: 10 negative, 5 neutral, 4 positive, 3 mixed) |
| 3 | Seeded random draw within each group (seed 2050) | 97 → 50 |

- **Measured on the baseline only.** Film-period SCR density is the outcome of H1–H3, so selecting
  on it would build the result into the selection. The baseline precedes every film, so the film
  group cannot cause an exclusion.
- **Model-free and set before deconvolution:** the measures were printed for all 119
  (`select-measures`), the rule was written into the code, then `select` ran.
- **No group difference at baseline among the 50** (Kruskal–Wallis, `data/selection_balance.csv`): level
  p = 0.90, SCR rate p = 0.13, median SCR p = 0.82, roughness p = 0.14.
- Measured but not used: jumps (none left after stage 1), slow noise (no quiet baseline stretch in
  48 of 119), SNR (high for nearly all; median 48).

## Part 7: the first final run, and why the synthetic data were rebuilt

`own4_p1000_k14_f3` ran once on the 50 real participants (P) and on two test sessions per
participant from the first generator: F (flat noise, as in tuning) and T (with slow noise). 150
jobs (`data/own_final_test_scores.csv`; outputs kept in `hpc/own/final/output/own4_p1000_k14_f3`).
These test sessions replaced the 18 S test sessions of Part 4 (never run): one per selected
participant tests the setting on the people it will be applied to.

| Test sessions | F1 visible | Recall visible | Precision | Decay error |
|---|---|---|---|---|
| F, flat noise (50) | 0.87 | 0.91 | 0.90 | +3% |
| T, slow noise (50) | 0.83 | 0.90 | 0.79 | +1% |

The scores held, but the real run showed the synthetic sessions were not like the real inputs in
a way that matters to the solver. **GCV's λ2 stopped at its limit in 91% of real windows against
38% in tuning**: the penalties do not grow with the signal as the fit does (sparsity ∝ size^0.5,
prior ∝ 1, fit ∝ size²), so a solver tuned on sessions of the wrong scale is tuned for a different
balance. Settings tuned on unrealistic sessions cannot be defended on real ones, so the sessions
were rebuilt from the 50 selected participants and the tuning repeated (round 6).

## Part 8: the new synthetic sessions (frozen 2026-11-11)

**Goal:** realistic signals whose truth is known, so accuracy can be measured. They have to look
like the real participants in what affects deconvolution (scale, rates, sizes, noise, tonic). The
exact timing of a participant's responses does not matter. Code: `generator.flat_session`,
`tbme_synth.build_blocks`. Checks: `synth-check` (`data/synth50_check.csv`). Visual notes:
`results/figures/tune50_preview/INSPECTION.md`.

**The design** (per participant, nothing fitted across participants):

1. The participant's real timeline (baseline and films).
2. Their tonic: 5th percentile over 60 s, smoothed over 60 s.
3. Their counted responses (peaks ≥ 0.02 µS on the real signal), reused per run: each film keeps
   its own rate and its own sizes, each response used once in shuffled order, at random times at
   least 1 s apart. Sizes jittered 5% (log), capped at their largest SCR.
4. One shape per session, drawn at random within physiological limits: peak 1–3 s after onset,
   half-recovery 2–10 s, decay ≥ 2× rise, within the solver's bounds.
5. Slow fluctuations (0.1–0.5 Hz), matched on the solver's input to their count of small SCRs
   (0.02–0.05 µS); then white noise up to their roughness, then quantisation.

**How it got there.** Each version was inspected page by page against its real participant.

| Version | Problem found | Decision |
|---|---|---|
| Cohort calibration (`synth-calibrate`: rate ×1.45, size ×0.70, spread ×1.06, noise ×0.81) | Iterative factors ran away (v3 fitted means, v4 hit the bounds); film responses the same size as baseline ones; big film-onset responses missing; needle-like shapes | Dropped: factors fitted across participants overfit, and the generator kept growing. Replaced by the plug-in above |
| Plug-in, shapes from another participant's solver fits | Shapes unlike the participant; the pool held odd fits (decay < 1.5× rise in 87 of 417 windows); activity flat across films | Pool filtered; one rate per film |
| Responses drawn with replacement per block | A participant's large responses landed in the wrong film (U118's 1.98 µS response) | Each run reuses its own responses, once each |
| Shapes from solver fits | Copying shapes is out of scope and inherits the solver's errors: the question is whether the solver recovers a known shape | Shapes drawn at random within physiological limits; tuning set stratified 6 fast / 6 typical / 6 slow |
| Fluctuations matched on the plain 2 Hz signal | Typical size on the solver's input 0.067 µS against the real 0.094 | Matched on the solver's input: 0.099 against 0.094 |

**Final check** (generated / real, `synth-check`): roughness 1.02, small SCRs 1.00, median size
1.05, p90 size 1.11, block sizes 1.14 / 1.14, clustering 0.95, all counted 0.87, large (≥ 0.05 µS)
0.81; on the solver's input: small SCRs 1.01, typical size 0.90, roughness 1.04. A slow-noise
variant (`own4s`) added nothing (slow power missing 1.3·10⁻⁷ µS²) and was dropped.

**Considered and rejected:**
- **Copying each participant's response times** (anchoring): timing is irrelevant to
  deconvolution accuracy and ties the truth to one participant's history.
- **Re-synthesis as the main truth:** rebuilding each real signal from the solver's own output is
  what the papers do (Amin & Faghih 2019, 2021), but it is circular. The truth is then the solver's
  own pulses: pulses it missed are missing from the truth, and its biases are built in. It is kept
  as a secondary comparison with the literature (Part 10).

**Limitations, reported not fixed:** tonic ramps at large onset events (a sharp rise followed by
minutes of plateau becomes a ramp under any windowed tonic); large responses about 20% fewer and
larger (counted responses are already merged and merge again); real bursts at specific moments
are not timed; the generator and the solver share the same response model.

### Round 6: scale (3 settings + 2 scale checks, 18 new tuning sessions U, 90 jobs)

A **first round-6 run** on an earlier version of these sessions (`hpc/own/tune_pre`) cannot be
scored: its truth files were overwritten when the sessions were regenerated, and the old code was
not committed. Its outputs are kept but not used. Round 6 was run again on the frozen sessions.

| Setting | F1 visible | Recall | Precision | Note |
|---|---|---|---|---|
| **A** `own4_p1000_k14_f3_s8` (round 5 + 8 random starts, the paper's number) | **0.835** | 0.850 | 0.816 | |
| B: A with the GCV sparsity limit 10× higher | 0.827 | 0.850 | 0.819 | λ2 at its limit in 0% of windows against 45%: the limit costs nothing |
| C: A with each session scaled to a typical response size | 0.792 | 0.836 | 0.806 | worse: noise is about constant in µS, so scaling small responders up amplifies it |
| A with y × 0.5 | 0.815 | 0.776 | 0.883 | |
| A with y × 2 | 0.784 | 0.900 | 0.729 | |

- **Scale shifts recall against precision**, and ×1 is best. So no normalisation: the solver runs
  in µS.
- Slow shapes (half-recovery > 7 s): decay about 40% short.
- **Frozen: A, `own4_p1000_k14_f3_s8`.**

## Part 9: the final run (P + V, 100 jobs)

The frozen setting ran once on the 50 real participants (P) and on 50 test sessions (V), one per
participant, made by the frozen generator with its own seed (8080), never used in tuning
(`data/own_test50_scores.csv`). Figures: `results/figures/test50_deconv.pdf` and
`real50_deconv.pdf` (signal, fit, tonic, true / found / missed / false pulses).

| Test sessions | n | F1 visible | Recall visible | Precision | Rise error | Decay error |
|---|---|---|---|---|---|---|
| fast | 9 | 0.83 | 0.87 | 0.86 | −6% | +12% |
| typical | 21 | 0.84 | 0.83 | 0.89 | +6% | −6% |
| slow | 20 | 0.79 | 0.85 | 0.74 | +17% | −37% |
| **all** | 50 | **0.82** | **0.85** | **0.85** | +9% | −13% |

Timing error 0.16 s, amplitude rank correlation 0.98. Median signed errors.

**Where the false pulses come from** (15% of returned pulses in the median session; read off the
figures):
1. **A rigid tonic:** slow rises the tonic cannot follow are filled with pulses.
2. **Long decays fitted short:** the recovery's tail is left over and filled with pulses. The
   prior pulls decays toward 2.77 s, and the tonic takes part of the tails.
3. Doublets on single responses, window joins, small fluctuations.

1 and 2 are linked through the knots, the prior and the constraint tonic ≤ signal.

**Real participants:** rise 1.17 ± 0.38 s, decay 3.38 ± 1.45 s (481 windows by the authors'
rule). By the half-recovery of each window's fitted shape: 67% fast, 32% typical, 0% slow
(`data/real_shape_classes.csv`). On the test sessions slow shapes come out 37% short, so the
real fits under-count slow shapes by an unknown amount.

## Part 10: re-synthesis (the papers' test)

Each real signal rebuilt from our final-run deconvolution (R), plus white noise at that
participant's own fit residual: SD median 0.0144 µS (0.0086–0.0383), 25 dB against the phasic part
(21–30). The truth is our pulses: median 230 per session, height median 0.103 µS. The fit of the
real signals: R² median 0.999 (`data/resynth50_cohort.csv`). A first version used the papers' fixed
25 dB. It was replaced: we follow the papers' method, not their numbers, on a different dataset.
Deconvolved again with the frozen setting (50 jobs). Circular by construction (Part 8), so it is
a comparison with the literature, not a measure of accuracy (`data/own_resynth50_scores.csv`,
`results/figures/resynth50_deconv.pdf`).

| | F1 visible | Recall visible | Precision | Timing | Amplitude ρ | Rise error | Decay error | R² |
|---|---|---|---|---|---|---|---|---|
| 49 sessions (R115 pending) | 0.859 | 0.894 | 0.838 | 0.15 s | 0.92 | −12% | +26% | 0.995 |
| by film group | 0.84–0.86 | 0.86–0.91 | 0.83–0.85 | 0.14–0.16 s | 0.91–0.93 | −10 to −19% | +24 to +29% | 0.994–0.996 |

- **No film group is deconvolved better than another**, so accuracy cannot drive a group
  difference in H1–H4.
- **The solver does not return its own decays:** rebuilt from its own shapes, they come back 26%
  longer. With the fast test sessions (+12%, Part 9) and the waveform measurement (solver ×1.19,
  Part 11), three independent checks agree: on fast and typical shapes, which most real
  participants have, the solver's decays run long. The real decays (median 3.4 s) are likely
  overestimated; reported as a limitation.
- R115's first job ran out of memory while starting MATLAB's workers; it was resubmitted at 16 GB.

## Part 11: the prior on τ

**The question:** on the test sessions, slow shapes lose precision to false pulses along long
recoveries fitted short. Is the prior the cause?

**The paper's prior** (0.65 ± 0.21 s rise, 2.77 ± 0.52 s decay) is a normal fit to the estimates of
the authors' previous study (Amin & Faghih 2019: auditory stimulation, 26 participants). On our
real 50, 47% of the frozen run's windows fall outside its ±2 SD (`data/decay_vs_prior.csv`).

| Option | Decision | Why |
|---|---|---|
| Refit the prior to our own estimates on the real 50 (rise 1.17 ± 0.38, decay 3.38 ± 1.45 s; `data/prior_from_real50.csv`) | dropped | Circular: it feeds the solver's own biased outputs back into it |
| A hierarchical (Bayesian) prior | dropped | Too much machinery for the question |
| A prior measured on the waveforms with no model (`scr-shapes`) | dropped | Failed the check set before the run (below) |
| **No prior, only the bounds** | **round 7** | Matches the generator, which draws shapes uniformly within physiological limits |

**The waveform measurement** (`src/eda/scr_shapes.py`). On every isolated SCR ≥ 0.05 µS: the time
from the 50% point on the way up to the peak (half-rise) and from the peak to the 50% point on the
way down (half-recovery). Then the response shape with the same two times. Checked first on the
test sessions, where the true shape is known. The pass rule, set before the run: median error
within ±10% and Spearman ≥ 0.8 across sessions with ≥ 5 SCRs. Recovery and decay are required.

| | Half-rise | Half-recovery | Rise τ | Decay τ |
|---|---|---|---|---|
| Error on the test sessions (40) | −0.1% | +12.9% | −12.3% | +13.9% |
| Spearman with the truth | 0.97 | 0.93 | 0.84 | 0.91 |
| Verdict | pass | **fail** | fail | **fail** |
| Real 50, median (444 SCRs) | 1.07 s | 2.49 s | 0.64 s | 2.05 s |

- A first version timed the rise from the onset (the lowest point before the peak). It came out
  43% long, because the onset sits in a flat, noisy stretch. It was replaced by the half-amplitude
  times, which sit where the signal is steep.
- **36% of real isolated SCRs recover as fast as they rise** (median 1.36 s down, 1.29 s up). No
  response shape of the model can produce that, so their decay sits at the 1.5 s bound. The test
  sessions do not show it, so it is a property of the real signals that the generator lacks. They
  could be breathing-paced fluctuations or fast-recovering SCRs; the waveform cannot tell.
- **Against the solver's fits** (42 participants with ≥ 5 SCRs): the solver is longer (half-rise
  ×1.10, Spearman 0.61; half-recovery ×1.19, Spearman 0.58). On the test sessions the measurement
  comes out longer than the solver in every shape class. The real waveforms recover faster than
  the solver says, and closer to the paper's prior (half-rise 0.96 s, half-recovery about 2.6 s).
- **Stopped here.** Refining the measurement until it passes would tune it to the check, the same
  trap as the generator's calibration factors. Kept as a description only
  (`data/prior_measured50.csv`, `scr_shapes_real50.csv`, `scr_shapes_validation.csv`).

### Round 7: no prior (1 setting, 18 jobs, pending)

| Setting | Against |
|---|---|
| `own4_p1000_k14_f3_s8_free`: A with no prior on rise or decay, only the bounds | A (round 6, same 18 U sessions) |

If the two score the same, the prior does not matter for detection, and A stays. If one wins, it
goes to the 50 V test sessions and the real 50 without further tuning. Its generalisation is the
justification: tuned on shapes drawn uniformly within physiological limits, it is not tied to one
range of shapes.

## Current status

**Frozen setting: `own4_p1000_k14_f3_s8`** (Part 9). 4 Hz input with a 1 Hz low-pass; prior weight
2·10⁻⁴ (the paper's 0.1 weakened 1000×, doubled at 4 Hz); knots every 14 s; phase-1 λ limit 2·10⁻³;
GCV limits 2·10⁻⁴ (λ2) and 2·10⁻⁴ (λ1); one impulse per signal sample; 8 random starts; in µS.
Everything else as in `own_config.m`. On the 50 unseen test sessions: F1 0.82, recall 0.85,
precision 0.85.

Limitations to report:

1. Slow responses (half-recovery > 7 s): decay 37% short and precision 0.74, from false pulses
   along their tails (Part 9).
2. False pulses where the tonic rises faster than its knots allow.
3. True rises under about 0.45 s overestimated, near what a 4 Hz signal resolves (round 3).
4. The generator's limits (Part 8), and the real signals' fast symmetric transients it lacks
   (Part 11).
5. Generator and solver share the same response model, which favours model-based methods in any
   comparison on synthetic data.
6. The first round-6 run could not be scored (its truth was overwritten).

## Next steps

1. **Round 7** (no prior against A) and its decision.
2. **Score the re-synthesis** (Part 10).
3. **Check that the real pulses make sense before stage 2**, blind to the hypotheses: plausible
   resting rates, pulses under the visible SCRs, none in flat stretches, sizes against the counted
   SCRs, plausible shapes per participant. The settings are frozen, so this cannot feed back into
   tuning, and H1–H6 are not looked at.
4. **Compare with other methods** under the same conditions (the V test sessions and the real 50,
   same input, same scoring): cvxEDA and the authors' published solver (branch `black-box`), each
   at its published defaults or tuned the same way, stated. On real data only fit, sparsity and
   plausibility can be compared, since there is no truth.
5. Stage 2 (impulses → arousal state) and hypotheses H1–H6, with a check at ≥ 0.05 µS. That code
   is on `black-box` (`state.py`) and comes over then.
6. Report and slides.

## How to run a round

1. Local: export the inputs (`own4-export`), set the round's settings in
   `src/own/make_own_bundle.sh`, run it. It writes `hpc/own_bundle.zip`.
2. Upload the zip to a new folder per run, `~/Project/own/round<k>/` on Torch. There:
   `unzip -oq own_bundle.zip`, `cd own_bundle`, `sbatch --mem=8GB run_own.sbatch` (8 GB keeps the
   array under the per-user memory limit). One job per session and setting.
3. On Torch, `tar -czf output_round<k>.tar.gz output` (zip is not installed). Download it, back up
   `data/own_tune_scores.csv`, extract it into `hpc/own/tune/`, then
   `PYTHONPATH=src python -m eda.run own-score`. It scores every setting found there and rewrites
   `data/own_tune_scores.csv`.

Before any code change: `own_test` in MATLAB (`addpath src/own; own_test`).

## Where everything is

| What | Where |
|---|---|
| **Stage 2's input:** every real pulse and window fit | `data/real50_pulses.csv`, `data/real50_windows.csv` (`real-pulses`) |
| Code, one module per stage | README, Layout; every command in `src/eda/run.py` |
| Scores of every tuning setting, per session (rounds 1–7) | `data/own_tune_scores.csv` |
| Cluster output and logs, every round | `hpc/own/tune/output/<tag>/`, `hpc/own/tune/slurm-own-*.out`; unscorable first round-6 run: `hpc/own/tune_pre/` |
| Settings of each round | `src/own/make_own_bundle.sh` (comments), `src/own/own_config.m` |
| Final runs: settings, outputs, scores | `src/own/make_own_final_bundle.sh`; `hpc/own/final/output/<tag>/`; `data/own_final_test_scores.csv` (first, F/T), `data/own_test50_scores.csv` (V) |
| Re-synthesis | `src/own/make_own_resynth_bundle.sh`, `data/resynth50_cohort.csv`, `data/own_resynth50_scores.csv`, `hpc/own/resynth/output/<tag>/` |
| Participant selection | `data/selection_measures.csv`, `data/selection.csv`, `data/selection_balance.csv` |
| Tuning / test split, first generator | `data/own_split.csv` |
| Truth of each synthetic input | `data/tbme_synth/<prefix><id>_<input>.npz` (prefixes S, F, T, U, V, R; see Terms) |
| Synthetic cohorts and checks | first generator: `data/tbme_synth_cohort.csv`, `data/tbme_synth_check.csv`; new: `data/tune50_cohort.csv`, `data/test50_cohort.csv`, `data/synth50_check.csv`, `data/own_scale.csv` |
| Prior investigation | `data/decay_vs_prior.csv`, `data/prior_from_real50.csv`, `data/real_shape_classes.csv`, `data/prior_measured50.csv`, `data/scr_shapes_real50.csv`, `data/scr_shapes_validation.csv` |
| Generator inspection notes | `results/figures/tune50_preview/INSPECTION.md`, `results/figures/tune50_preview_vs_real.pdf` |
| Deconvolution figures, final run and re-synthesis | `results/figures/test50_deconv.pdf`, `real50_deconv.pdf`, `resynth50_deconv.pdf` |
| Input checks | `data/input_check.csv`, `data/input_check_own4.csv`, `data/filter_response.csv`, `data/preprocess_stages.csv`, `data/raw_overview.csv` |
| Participant screen and profiles | `data/quality_study6.csv`, `data/profile_study6.csv`, `data/profile_study6_peaks.csv` |
| Figures | `results/figures/stages_*.png`, `raw_atlas_*.png`, `filter_response.png` |
