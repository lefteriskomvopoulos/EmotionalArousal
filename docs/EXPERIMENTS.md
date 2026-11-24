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
   both dropped. Round 7 tested no prior against the paper's prior, weak: no prior was no better
   and less stable, so the frozen setting stays.
9. **Re-synthesis** (Part 10), the papers' own test: F1 0.86, the same in every film group.
10. **Round 8, the tonic** (Part 11): wider knots trade fewer misses for more false pulses and
   leave slow sessions' tonic error unchanged. The error is a level offset during dense slow
   responding, where the true floor is never seen, so no spline setting fixes it; the frozen
   setting stays and it is reported as a limit.

11. **Stage 2** (Part 13): the Data Challenge 2 filters, validated on 400 synthetic sessions (one
   criterion rescored after the result, flagged), run on the 50.
12. **H1–H6** (Part 14), set in writing first and then run once.
    - **H2 supported:** emotional films are above neutral, mostly because neutral films fall below
      baseline.
    - **H1 not supported.**
    - **H3 not supported:** the negative group is not above neutral. Within the mixed group,
      negative and positive films are equivalent.
    - **H4 not supported.**
    - **H5:** no evidence of separation.
    - **H6:** not supported (r +0.14, p 0.074; run 2026-11-23 once the raw files were in, Part 19).
13. **The tonic level as a third observation** (Parts 15–16, exploratory variants; both failed
    validation and were run on the 50 at Piyush's request).
    - **Unweighted, `tonic14`:** the tonic level took over the state, with overconfident bands and
      drift leaking in. H1 and H2 flipped in the predicted direction.
    - **With a balanced floor, `tonic14b`:** the overconfidence is fixed and the drift leak is
      smaller, but it is not gone. The estimates are back near the main analysis's, with H2 at
      p 0.09.
    - **Neither changes a verdict.**
    - **Differenced, `tonic14d` (Part 17):** the tonic level's changes, with the floor. Drift is
      handled (its drift scenario passes), but the clean scenario fails narrowly on a stretched
      scale (bias 0.13 against a limit of 0.1). Not run on the 50.
    - **Calibrated, `tonic14c` (Part 18):** the link is calibrated from the pulse-only state
      (regression calibration). It **passes on fresh synthetic seeds**, the first tonic variant to
      pass. On the 50 every verdict matches the main analysis, the estimates agree to within 0.03,
      and the intervals are slightly narrower (H2 p 0.010). It is a validated robustness check,
      not a replacement for Part 14.
14. **The affect slider** (Part 19): H6 run as set in Part 14, plus the evaluation of Oliver et al.
    (2025), adapted.
    - The state does not identify the moments of highest self-reported intensity (chance).
    - It does follow the direction of changes slightly above chance (54–59%, p 0.002–0.05), the
      same level the paper reports.

Still to come: the rest of the check that the real pulses make sense, a comparison with other
methods, H6 if the raw files are obtained, and the report.

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

**At ≥ 0.05 µS** (the `_05` columns of `tbme_score`: true pulses ≥ 0.05 µS found by any returned
pulse within 1 s, returned pulses ≥ 0.05 µS matching any true one): F1 **0.96**, recall 0.98,
precision 0.96; fast 0.98, typical 0.97, slow 0.94; film groups 0.95–0.97. Almost all errors are
pulses of 0.02–0.05 µS, at the noise floor.

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
| 50 sessions | 0.860 | 0.892 | 0.839 | 0.15 s | 0.92 | −14% | +26% | 0.995 |
| by film group | 0.84–0.88 | 0.86–0.91 | 0.83–0.86 | 0.15–0.16 s | 0.91–0.93 | −10 to −19% | +23 to +29% | 0.994–0.996 |

At ≥ 0.05 µS: F1 0.96, recall 0.95, precision 0.97 (film groups 0.93–0.96).

- **No film group is deconvolved better than another**, so accuracy cannot drive a group
  difference in H1–H4.
- **The solver does not return its own decays:** rebuilt from its own shapes, they come back 26%
  longer. With the fast test sessions (+12%, Part 9) and the waveform measurement (solver ×1.19,
  Part 11), three independent checks agree: on fast and typical shapes, which most real
  participants have, the solver's decays run long. The real decays (median 3.4 s) are likely
  overestimated; reported as a limitation.
- R115's first job ran out of memory while starting MATLAB's workers; it was resubmitted at 16 GB
  and scored with the rest (the 49-session scores were the same to within 0.002 in F1).

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
| No prior, only the bounds | tested in round 7, not adopted | Matches the generator, which draws shapes uniformly within physiological limits; no better and less stable (below) |

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

### Round 7: no prior (1 setting, 18 jobs)

`own4_p1000_k14_f3_s8_free`: A with no prior on rise or decay, only the bounds, against A (round
6) on the same 18 U sessions. The rule, set before the run: if the two score the same, the prior
does not matter for detection and A stays; if one wins, it goes to V and the real 50 without
further tuning.

| Median | A (paper's prior, weak) | No prior |
|---|---|---|
| F1 visible, all 18 | **0.835** | 0.825 |
| Recall / precision | 0.850 / 0.816 | 0.853 / 0.812 |
| Amplitude rank correlation | **0.971** | 0.947 |
| Decay error, all 18 | −0.8% | +6.3% |
| Windows meeting the authors' rule | **1.00** | 0.83 |
| Slow shapes: F1 / precision | **0.828 / 0.767** | 0.794 / 0.736 |
| Slow shapes: decay error | −41% | −28% |
| Slow shapes: windows meeting the rule | **0.83** | 0.36 |
| Typical shapes: decay error | **−0.8%** | +10% |

- **No prior is no better for detection** (F1 within 0.01 overall), and worse on what the prior
  was there for: on slow shapes, two thirds of the windows lose the selection rule (τ rise at a
  bound), as in round 4, and precision falls.
- **The prior is not what makes the false pulses on slow recoveries.** Without it the slow decays
  come out less short (−28% against −41%), but precision drops instead of recovering. What is left
  is the tonic taking part of the long tails (Part 9, cause 2 and the knots).
- **A stays.** Nothing changes for the V test sessions or the real 50, so nothing is re-run.

### Round 8: the tonic's flexibility (2 settings, 36 jobs)

**The question:** with the prior cleared (round 7), is the tonic what takes the slow tails? In A,
GCV's tonic penalty λ1 is at its limit in only 7–12% of the U windows and chosen near zero
(10⁻⁸–10⁻⁶ on P53), so the tonic's shape is set by the knot spacing alone. On slow shapes it takes
18% of each response (`phasic_taken` 0.18, about 0 on fast and typical): a spline with knots every
14 s can bend into a 10 s recovery. Round 5 tested the spacing on the first 18 (S) only.

| Setting | Change from A |
|---|---|
| `own4_p1000_k20_f3_s8` | knots every 20 s |
| `own4_p1000_k28_f3_s8` | knots every 28 s, twice A's |

**Rule, set before the run:** a variant replaces A only if slow-shape precision and `phasic_taken`
both improve, and F1 visible on fast and typical shapes falls by no more than 0.01 each. Otherwise
A stays. A stiffer tonic may follow fast tonic rises worse (Part 9, cause 1); the F1 condition
covers that. A variant that passes runs once on V and the real 50 (100 jobs, no further tuning),
and the stage-2 table is rebuilt from it.

**Considered and dropped:** a floor on λ1. The penalty is a ridge on the spline coefficients,
which are the tonic's level in µS. A floor pulls the tonic down under the constraint tonic ≤
signal and hands more of the signal to pulses; it does not stiffen the shape. If neither spacing
passed, the next candidate was a penalty on the tonic's curvature (a P-spline, Eilers & Marx
1996), a change to the paper's method.

**Result** (`own-score`; `round-compare`: `data/round8_compare.csv`,
`results/figures/round8_tonic.pdf`, one page per session with the true tonic dotted):

| Median, 18 U sessions | A (14 s) | 20 s | 28 s |
|---|---|---|---|
| F1 visible | **0.835** | 0.803 | 0.794 |
| Recall / precision | 0.850 / **0.816** | 0.887 / 0.786 | **0.900** / 0.760 |
| Slow: F1 / precision | **0.828 / 0.767** | 0.784 / 0.722 | 0.738 / 0.631 |
| Slow: tonic's share of each response | 0.184 | 0.198 | 0.185 |
| Fast: F1 | **0.895** | 0.877 | 0.840 |
| Typical: F1 | 0.771 | 0.795 | **0.800** |

| Summed over the 18 | A (14 s) | 20 s | 28 s |
|---|---|---|---|
| Visible true pulses missed | 395 | 336 | 327 |
| False pulses (of returned) | 452 of 2292 | 567 of 2466 | 739 of 2647 |
| False, no true pulse within 2 s | 333 | 447 | 594 |
| False on slow sessions (≥ 0.05 µS) | 213 (57) | 293 (106) | 418 (126) |

- **Both fail the rule:** slow precision falls, the tonic's share on slow sessions does not move,
  fast F1 drops more than 0.01. A stays; nothing is re-run.
- **A stiffer tonic trades misses for false pulses.** It misses 59–68 fewer true pulses (mostly
  small ones the flexible tonic absorbed; typical F1 +0.02–0.03), but returns 115–287 more false
  ones, nearly all standing alone: real tonic changes the stiffer spline cannot follow are filled
  with pulses (Part 9, cause 1).
- **The slow-session error is an offset, not a bend.** In the figures (U12, 13–18 min), the
  estimated tonic sits 0.2–0.4 µS above the true one at every spacing. Overlapping long recoveries
  never let the signal return to the true floor, so the data hold no information about it there:
  a higher tonic with shorter decays fits about as well as the truth and needs smaller pulses,
  which the sparsity term favours (our reading of the figures and counts).
  This is why slow decays come out short (−37%) with or without the prior (round 7), and why no
  spline setting changes it. A curvature penalty would not either (the error is smooth), so it
  is not run. Reported as a limit of separating tonic from phasic during dense slow responding,
  a regime rare in our real data (Parts 9 and 11: real fits and waveforms are mostly fast or
  typical).
- **Why GCV leaves the tonic unpenalised:** the penalty λ1‖q‖² acts on the spline coefficients,
  which are the tonic's level (about 1–3 µS). A 200 s window has only about 20 of them against 800
  samples, so shrinking them barely reduces the tonic's effective parameters while any real λ1 pulls
  the whole tonic toward zero. GCV therefore picks λ1 at the low end of its search; the knots alone
  set the tonic's flexibility.

**Other ways to treat the tonic, checked against the sources (2026-11-14), and why none is run:**

| Method | Tonic | Why not tested here |
|---|---|---|
| cvxEDA (Greco et al. 2016; official code) | spline with knots every 10 s, plus an unpenalised offset and linear trend; ridge ½γ‖l‖² on the spline, γ = 0.01 fixed; joint with a sparse non-negative driver | inside a 200 s window within a dense slow stretch the offset rises for free, so it cannot see the floor either; compared as a whole method instead (Next steps) |
| Ledalab CDA (Benedek & Kaernbach 2010) | from the deconvolved driver between detected impulses, averaged on a 10 s grid (100 s in their DDA), cubic-spline interpolated | anchors the floor on quiet moments minutes away, which fails when the tonic truly changes during a dense stretch; and our generator's true tonic is itself a low-percentile floor, so a floor-type estimator would be scored against a floor-type truth (partly circular) |
| The Faghih group's earlier multichannel method (as described by Alam, Amin & Faghih 2023) | removed from each channel before deconvolution | same as above |
| Amin & Faghih 2022 (PLOS Comput Biol) | a state of a three-state state-space model, no spline | a different solver, not a tonic option |
| NeuroKit2 default (`eda_phasic`) | Butterworth low-pass at 0.05 Hz | filtering, no deconvolution |

Benedek & Kaernbach note that responses have "a slowly recovering trail which may obscure any
tonic SC activity": none of these methods can recover the floor where it is never visible.

## Part 12: are the real pulse counts plausible? (blind to the hypotheses)

**The question:** the frozen setting returns 11.7 pulses per minute on the real 50 (median; 9.4 of
them ≥ 0.05 µS), 1.87× the counted peaks of the same recordings (7.0 per minute ≥ 0.02 µS,
`data/profile_study6.csv`). At rest (baseline): 13.0 per minute, 10.1 ≥ 0.05 µS, against 6.6
counted. Published resting rates of non-specific responses: 1–3 per minute at 0.05 µS (Dawson et
al. 2017, as cited by Kurinec et al. 2022), about 5 and up to 10 per minute at 0.01–0.05 µS
(Zimmer 2000, as cited by Gertler et al. 2020), 6.7 per minute above 0.1 µS in resting adolescents
(Gertler et al. 2020). Counts of responses merge overlapping ones; deconvolution separates them,
so some excess is expected. How much is false?

**The check** (`pulse-check`, `data/pulse_check_isolated.csv`): the pulses returned under each
isolated SCR (Part 11's measurement: no other peak ≥ 0.02 µS from 5 s before onset to
half-recovery), on its rise (1 s before onset to the peak) and on its recovery (peak to
half-recovery). The same on the 50 test sessions, rebuilt exactly, where the true pulses in the
same windows are counted too. Film groups and blocks are not looked at.

| Mean per isolated SCR | Test: true | Test: returned | Real: returned |
|---|---|---|---|
| Rise, all | 1.23 | 1.24 | **1.44** |
| Rise, ≥ 0.05 µS | 1.13 | 1.18 | 1.27 |
| Recovery, all | 0.17 | 0.33 | 0.31 |
| Recovery, ≥ 0.05 µS | – | 0.11 | 0.18 |
| Rises with 1 / 2 / 3+ pulses | 79 / 19 / 2% | 78 / 20 / 2% | 62 / 26 / 10% |

452 isolated SCRs on the test sessions, 444 on the real 50 (amplitude median 0.35 and 0.20 µS).

- **Where the truth is known the solver counts isolated responses right** (1.24 against 1.23 per
  rise); its extra is small pulses on recoveries (0.33 against 0.17).
- **On real data it adds about 0.2 pulses per rise (+16%), mostly small** (+8% at ≥ 0.05 µS):
  doublets on response rises (36% of real rises get two or more pulses, 22% on test), not pulses
  in quiet stretches.
- **That accounts for the excess over counted peaks.** On the test sessions counting finds 0.74 of
  the true pulses (overlapping responses merge; `synth-check` rate 0.74) and the solver returns
  1.03 per true pulse (round 6). So 7.0 counted peaks per minute imply about 9.7 pulses per minute
  if the solver behaved on real data as on synthetic; it returns 11.7, about 20% more, the same
  size as the doublets on isolated rises.
- **The doublets cannot be classified from the data:** real responses built from more than one
  burst, or the solver splitting real responses whose shape the model cannot produce (36% of real
  isolated SCRs recover as fast as they rise, Part 11).
- **Verdict:** the real pulse counts are plausible for a deconvolution, about 20% high, the excess
  small doublets on response rises. Reported as a limitation; stage 2's main threshold of 0.05 µS
  removes about half of it.

**Further checks (2026-11-14):**

- **Spacing.** Pulses closer than 1 s are pruned to the largest (`sep_s = 1`, the paper's rule,
  in phase 1; phase 2 cannot add pulse positions). It holds: none of the 12,221 real pulses is
  within 1 s of another. 26% of real pulses come 1–2 s after the previous one, against 13% on the
  test sessions: that is where the doublets are. A 2 s rule would also merge true bursts 1–2 s
  apart, and at 0.05 µS the excess is only +8%, so it is not pursued.
- **Sparser is not more correct.** Sparser settings were tuned against the truth and lost: round 6
  with the input halved (precision 0.88, recall 0.78, F1 0.815 against 0.835) and round 1's strict
  sparsity limit (precision 1.00, recall 0.41). The frozen setting returns about as many pulses as
  are true (2,292 against 2,235 on the U sessions; 1.24 against 1.23 per isolated response).
- **Each person's pulse rate tracks their visible responding:** rank correlation with their counted
  response rate 0.73 (all pulses), 0.67 (≥ 0.05 µS), over the 50.
- **Other methods on the same data** (`black-box` branch tables): the authors' published solver on
  the four participants both runs share returns 6.3–11.9 pulses per minute (ours 8.1–17.2; counted
  responses 2.3–9.2), from 1.2 to 2.8× the counted rate (ours 1.7–3.5×). cvxEDA at its defaults on
  eight other participants: 11.4 per minute at rest, median size 0.038 µS.
- **No reference range exists for deconvolved pulses.** Published norms count responses, which merge
  overlapping bursts. At rest our 50 show 4.6 counted responses ≥ 0.05 µS per minute (1.2–9.6;
  `data/profile_study6_peaks.csv`), 1.5–4× the usual 1–3: an active sample, selected as clear
  responders (median baseline response ≥ 0.05 µS). Their pulse rate at ≥ 0.05 µS is 10.1 per minute
  at rest. What can be claimed: accurate where the truth is known, consistent with each person's
  visible responding, and in line with other deconvolution methods; not "within the physiological
  range" in the textbook sense.

## Part 13: stage 2, the state estimator (`src/stage2/`)

**The filters.** Stage 2 uses the state-space filters of the course's Data Challenge 2 (Piyush's
submission): `filter_one_mpp.m` (marked point process: pulse occurrence and size, Wickramasuriya &
Faghih, IEEE Access 2020) and `filter_one_bin.m` (pulse occurrence only). They are copied into
`s2_filter_mpp.m` and `s2_filter_bin.m` with the arithmetic unchanged: only the interface changed
(n and r passed separately, silent, a stop reason returned, the p_k confidence limits left out).
`s2_test_dc` runs both on the Data Challenge's own subjects with its starting values and
reproduces its saved results exactly (Subjects 1, 2, 8, 10; difference 0 in state, variance and
parameters, including Subject 2's MPP stop on r1 ≤ 0). These filters replace `state.py` on
`black-box`, whose fixed size noise and capped slope the proposal named as unvalidated.

**The input** (`s2_observations`): pulses ≥ 0.05 µS (rule 8) in 1 s bins over the whole segment
(baseline, transition, films), size r = log(height in µS) as in the proposal. Chosen on structure
alone, before any state was computed: the solver keeps pulses ≥ 1 s apart, so a 1 s bin holds at
most one pulse, where 5 s bins (the proposal) dropped a pulse in 37% of occupied bins. Two
exceptions to the 1 s spacing were found, contradicting Part 12's "none within 1 s": P60 at
1042.1 s and P80 at 376.0 s, pairs 0.25 s apart (one 4 Hz sample) in the same window. Adjacent
samples are one burst (proposal), so each pair is merged into one pulse with the summed height.
The 50: 1,116–1,357 bins each, 9,509 pulses, median 0.157 pulses per second, log height mean −1.86,
SD 0.74 (pooled over all blocks and participants).

**Starting values** (MPP): the rules of Data Challenge Problem 3F, from the data rather than the
cortisol-scale constants: r0 = mean(r), r1 = std(r), vr = var(r)/2 over the pulse bins,
ve = 5·10⁻³. Binary: ve = 0.05 as in the Data Challenge.

**Which filter (fixed before any real state was computed).** The MPP filter is the main one. When
it stops without converging (r1 ≤ 0, i.e. pulse sizes do not rise with the state, or the
iteration limit), that participant's state comes from the binary filter, as in Data Challenge
Subjects 2 and 8. The binary filter is also run on everyone as the timing-only check.

**Synthetic validation** (`s2_synth`; set before it was run). Each session copies one of the 50
real layouts (bin count and block codes only, no pulses). True state: 0 in the baseline and
transition; each film clip at Δ + c, c ~ N(0, 0.2²); plus a random walk with variance 2·10⁻⁴ per
second. Pulses from P(n=1) = 1/(1+exp(−(b0 + x))), b0 = logit(0.157); sizes r = −1.86 + r1·x +
N(0, 0.5). Δ ∈ {−0.5, 0, 0.5, 1.0} (logit units; 1.0 roughly doubles the pulse rate),
r1 ∈ {0, 0.5}: r1 = 0 is the Data Challenge's failure case, sizes that carry no state. 50 layouts ×
8 conditions = 400 sessions, fixed seeds. The contrast scored is the one H1–H4 use:
Δ̂ = mean state over film bins − mean over baseline bins, against the same on the true state.

Pass criteria, for the filter chosen by the rule above:

| | Criterion | Pass |
|---|---|---|
| C1 | Contrast recovery over all 400 sessions | r(Δ̂, Δ_true) ≥ 0.8 and slope 0.7–1.3 |
| C2 | Bias in the contrast, in each Δ level | |mean(Δ̂ − Δ_true)| ≤ 0.1 |
| C3 | Coverage of the true state by the 95% band | median over sessions 0.85–0.99 |
| C4 | Chosen filter converged | ≥ 95% of sessions |
| C5 | MPP usable when sizes do carry the state (r1 = 0.5) | MPP converges in ≥ 80% |

If C1–C4 pass, the 50 real participants are run with the rule above. If any fails, the real
participants are not run: the failure is recorded here and one change is made and tested.

**Result** (`data/stage2_synth_scores.csv`, 2026-11-16):

| | Result | |
|---|---|---|
| C1 | r(Δ̂, Δ_true) 0.963, slope 0.949 | pass |
| C2 | bias by Δ (−0.5, 0, 0.5, 1.0): +0.032, +0.030, +0.001, −0.029 | pass |
| C3 | coverage median 0.842 | **fail** |
| C4 | converged 1.000 (MPP 286, binary fallback 114) | pass |
| C5 | MPP converged at r1 = 0.5: 0.915 | pass |

- **The contrast H1–H4 test is recovered** by either filter: r 0.96–0.97, bias within ±0.03.
- **The MPP filter stops when sizes carry no state** (r1 = 0: converged in 52%), as in the Data
  Challenge, and the binary fallback takes over. At r1 = 0.5 it converged in 92%; 3 of the 34
  stops first blew up (r1 near 10⁸, state NaN), the Data Challenge's Subject 4 failure. The fallback
  covered all of them.
- **Why C3 failed** (diagnosis on 10 layouts per Δ at r1 = 0.5, same seeds; nothing tuned). Coverage
  is 0.96–0.98 at Δ = 0 and collapses at Δ = +1.0 (0.19 chosen, 0.45 binary). The filters pin b0 to
  the session's overall pulse log-odds; when the films raise the rate, b0 comes out higher and the
  whole state lower by the same amount. Mean(x̂ − x) against b̂0 − b0: +0.13 / −0.16 (Δ −0.5),
  −0.02 / −0.00 (0), −0.43 / +0.45 (0.5), −0.68 / +0.70 (1.0). With that constant removed, coverage
  is 0.98–1.00 at every Δ. Only b0 + x enters the model, so the absolute level is not identifiable
  by any setting of these filters; differences within a session are. C3 as written scored the
  absolute level.
- **Decision (2026-11-16, Piyush): C3 is rescored, and this is a change made after seeing the
  result.** C3 as first set failed, and that stays on record. The rescored criterion, C3′, scores
  what H1–H4 use, the state relative to the person's own baseline: x̂ − mean(x̂ over baseline
  bins) against x − mean(x over baseline bins), with the same 95% band half-width and the same
  pass range, median 0.85–0.99. It was written here before it was computed. The band does not
  include the uncertainty of the baseline mean, so C3′ is, if anything, strict. The rest of the
  run must come out identical (same seeds).
- **C3′ result: pass**, median 0.984 (by Δ: 0.989, 0.986, 0.989, 0.970; MPP and binary alike).
  The rerun reproduced the first run's 19 columns in all 400 rows exactly. The bands are slightly
  conservative (near the top of the range), not overconfident.

**The real 50** (`s2_run`, 2026-11-16): `data/stage2_params.csv` (stops and parameters),
`data/stage2_states_005.csv` (main) and `data/stage2_states_002.csv` (secondary), one row per
participant and 1 s bin: both filters' states, the chosen state with its 95% band, and p_k. The
checks below are blind to the hypotheses: no block or group contrast was computed.

- **0.05 µS (main):** MPP converged for 44 of 50; size rises with the state in all of them (r1
  0.09–1.33, median 0.46, near the synthetic 0.5). 6 stopped on r1 ≤ 0 (P15, P106, P111, P115,
  P168, P178; P111 after a blow-up) and use the binary state. Binary converged for all 50. No NaN
  in any chosen state. Median 95% band width 1.15. Converged MPP against binary state, per person:
  r median 0.92 (0.59–0.995).
- **0.02 µS (secondary):** MPP converged for 45; the other 5 blew up (3 stopped on r1 ≤ 0, 2 hit
  the iteration limit) and use the binary state.
- **Flag:** P68 has a large drift variance in both filters (ve 0.16 and 0.17 per s), so its state
  moves fast. Both filters agree, so it is the data, not a filter failure.
- `s2_plot(99)` → `results/figures/stage2_state_P99.png`: the method on one participant.
  `s2_plot_all` → `results/figures/stage2_states_real50.pdf`: all 50, one page each, ordered by film
  group then id. Each page title says whether the MPP state or the binary fallback is shown. The
  baseline is shaded; dotted lines mark the end of the baseline and each clip boundary (the stretch
  between the first two is the transition).

## Part 14: the H1–H6 tests (set 2026-11-16, before they were run)

Written after Part 13 and before any block or group contrast of the state was computed. Already
seen at this point: the descriptive look at pulse rates and sizes by block and film group (Next
steps, item 3, 2026-11-14). Where the proposal is specific, the plan follows its wording. Where a
choice had to be made after that look, it says so. The tests run once, in one script, and the
result is recorded whatever it is. Nothing in stage 1 or stage 2 (filters, choice rule, thresholds,
bins) changes after they are run.

**The design** (read from the state tables and the POPANE stimulus list,
`docs/dataset_check/metadata_R1.xlsx`; no outcome looked at). Every participant has a 300 s
baseline (marker −1), then a transition (left out), then six 120 s film clips in random order.
Neutral group: 101–106. Negative group: 201–206. Positive group: 301–306. Mixed group: 3 negative
and 3 positive clips from the same pools (checked for all 10). The earlier note "mixed group: 6
negative, 6 positive" (Next steps) was wrong and is corrected there. Clip emotions: fear 201,
anger 202–204, disgust 205, sadness 206, amusement 301–304, tenderness 305–306.

**Per-block summaries**, one row per participant and block, pulses ≥ 0.05 µS:

| | Measure, per block | Change score against the person's own baseline |
|---|---|---|
| S | mean chosen state over the block's bins | difference (logit units) |
| R | pulse rate: log((count + 0.5) / minutes); the 0.5 keeps an empty clip finite | difference (a log rate ratio) |
| Z | mean log height of the block's pulses (missing if the block has none) | difference |

- **A person's film period** is all bins of their six clips. The clips are equal in length, so this
  is the mean of the clip means. In the mixed group, the negative and positive periods are the bins
  of their three clips of each.
- **S is the primary measure, and the hypotheses are judged on it.** R and Z are reported next to
  every test, as the proposal says. Where S disagrees with R and Z, that is reported as a finding
  about the estimator.
- **Only within-person changes are used.** The state's absolute level is not identifiable (Part
  13), and the groups already differ at baseline.

**Statistics.**
- **Estimates:** each is a mean change with its 95% CI (t).
- **Tests:** Welch t-tests between groups, and one-sample or paired t-tests within a person. All are
  two-sided at α = 0.05. A directional hypothesis is supported only when p < 0.05 and the sign is
  the predicted one.
- **Equivalence:** by two one-sided tests (TOST). Two things count as equivalent when the 90% CI of
  their difference lies inside ±δ.
- **Rank-test check:** a rank test runs next to each t-test (Mann–Whitney between groups, Wilcoxon
  signed-rank within a person). Any disagreement is reported.
- **No correction across H1–H6.** Each is its own question in the proposal, and exact p values are
  reported.

**The equivalence margin is δ = 0.5 logit units.**
- **Where it comes from:** the smallest nonzero film effect in the stage-2 validation (Part 13).
  At the median rate of 0.157 pulses per second, that is about 1.5× the pulse rate. It is also 15×
  the filters' largest bias (0.032).
- **It was set after the descriptive look.** That look showed the mixed group's median rate higher
  during negative than positive films (9.8 against 7.2 per minute; log ratio 0.31). δ comes from
  the validation design, not from that number. The 90% CI is reported in full, so any other margin
  can be applied.
- **The result may be inconclusive.** With 10 people in the mixed group, the CI is likely to cross
  both 0 and ±δ. Such a result will be reported as inconclusive, not as support for either
  hypothesis.

| | Hypothesis (proposal) | Who, and the statistic | Test | Supported when |
|---|---|---|---|---|
| H1 | Films raise arousal above the resting baseline | All 50, own films; Δ = film − baseline | one-sample t | p < 0.05, Δ > 0 |
| H2 | Emotional films raise it more than neutral ones | Δ: negative, positive and mixed groups (30) against neutral (20) | Welch | p < 0.05, emotional > neutral |
| H3 (primary) | Arousal reflects intensity alone: negative and positive both exceed neutral, and do not differ from each other | (a) Δ, negative group (10) against neutral (20); (b) Δ, positive group (10) against neutral (20); (c) mixed group, D = Δneg − Δpos per person | (a), (b) Welch; (c) paired TOST | (a) and (b) p < 0.05 in the predicted direction, and (c) equivalent |
| H4 | Negative exceeds positive, within the same person | Mixed group, D | paired t | p < 0.05, D > 0 |
| H5 | Emotions within a valence do not separate | Clip changes, averaged per emotion. Negative group: fear, anger (mean of 3 clips), disgust, sadness. Positive group: amusement (mean of 4 clips) against tenderness (mean of 2 clips) | Friedman (4 negative emotions); paired t (positive) | p ≥ 0.05 in both, reported as no evidence of separation, not as equivalence |
| H6 | The decoded state tracks the participant's own affect slider | Everyone with slider data. Per person, Spearman r across the 6 clips between the clip's mean state and its mean slider intensity, \|a − 5.5\| | Fisher z, then one-sample t | p < 0.05, mean r > 0 |

- **H1 is the proposal's: all films, all 50.** It is also reported for each group, as a
  description. The descriptive look showed the neutral group's rate falling during films (0.77 of
  baseline), so H1 may fail through the neutral group. The definition predates that look and is
  kept.
- **H3 and H4 together.** If D is above 0 (p < 0.05) and its 90% CI lies inside ±δ, the verdict is
  "negative above positive, by less than the margin", and both results are given. The
  between-person version of (c), negative group against positive group with Welch TOST and the
  same δ, is reported as secondary: the proposal names the mixed group's within-person test as the
  one that tells H3 and H4 apart.
- **H5's limits.** Fear, disgust and sadness are each a single film, so emotion and film cannot be
  told apart, and a difference may come from the film. The mixed group is left out: each person
  saw 3 random clips per valence, so the emotions are unbalanced. Clip order is random, so position
  effects average out.
- **H6's details.**
  - The slider is valence, from 1 (extremely negative) to 10 (extremely positive) (Behnke et al.
    2022), so intensity is the distance from the midpoint, as in H3. The raw valence is reported
    alongside, two-sided.
  - Only the film clips are used: the paper does not say whether the slider was used during the
    baseline.
  - Anyone whose slider does not change across their six clips (r undefined) is left out, and the
    number is reported.
  - The slider is only in the raw POPANE files (`eda.dataset.load`, column `affect`). If they cannot
    be obtained by the deadline, H6 is not tested, and the report says so.

**Checks, run with the same tests and reported as secondary.** None of them can change a verdict.
1. The 0.02 µS threshold (`data/stage2_states_002.csv`).
2. The binary filter's state for all 50 (pulse timing only).
3. Without P68, flagged in Part 13 before any contrast.
4. The rank tests (above).

**Output.**
- **The script:** `src/stage2/s2_tests.m`.
- **What it writes:**
  - `data/stage2_block_summary.csv`;
  - `data/stage2_tests.csv`, one row per hypothesis, measure, threshold and check (n, estimate, CI,
    p, verdict);
  - `results/figures/stage2_hypotheses.png`, change scores by group for S, R and Z.
- **Before the real run,** it runs end to end once on a copy with the group labels and clip codes
  randomly permuted (fixed seed), so the code is tested without any real contrast being seen.

**The code check, as run** (`s2_tests(pwd, 'synthetic', true)`, 2026-11-16). It deviates from the
plan above, and the change is stricter. Permuting labels would still have shown the real H1, which
uses no labels. So the check used made-up states on the real layouts, with no real pulse or state
in them:
- **The made-up data:** +0.5 in every negative and positive clip, and 0 in neutral clips. Pulses
  and sizes come from the MPP model, and the state is the truth plus noise.
- **What came back:**
  - H1 estimate 0.27, against 0.30 expected (30 of the 50 see emotional films);
  - H3 (a) and (b) supported;
  - mixed-group D −0.003, equivalent; H4 not supported.
  - H5 came out "separation" at p 0.043, from the random-walk drift across clips. That is a false
    positive of the kind the 5% level allows.
- **H6's file reader** was tested only on a hand-made file in the format `eda.dataset.load`
  describes. It has not seen a real POPANE file.

**Result** (`data/stage2_tests.csv`, `data/stage2_block_summary.csv`,
`results/figures/stage2_hypotheses.png`; run once, 2026-11-16). All figures are for the main check,
the state (S), as change from each person's own baseline, in logit units. They are given as
estimate [95% CI], p (rank-test p).

| | Result | Verdict |
|---|---|---|
| H1 | All 50: −0.08 [−0.24, 0.07], p 0.30 (0.23). By group (descriptive): neutral −0.31 (p 0.03), negative −0.10, positive +0.19, mixed +0.11 | not supported |
| H2 | Emotional − neutral: +0.37 [0.05, 0.69], p 0.023 (0.019) | supported |
| H3 (a) | Negative − neutral: +0.20 [−0.21, 0.61], p 0.31 (0.39) | not supported |
| H3 (b) | Positive − neutral: +0.50 [0.08, 0.93], p 0.023 (0.011) | supported |
| H3 (c) | Mixed group, D = +0.05, 90% CI [−0.30, 0.41], inside ±0.5; TOST p 0.023 | equivalent |
| H3 | (a) fails | **not supported** |
| H4 | Mixed group, D = +0.05, p 0.78 (0.92) | not supported |
| H3/H4 | Within person, negative and positive films do not differ, and the difference is shown to be inside the margin | the H3 side |
| H3, secondary | Negative group − positive group: −0.30 [−0.75, 0.16]; 90% CI reaches −0.67 | not shown equivalent |
| H5 | Negative group, Friedman p 0.062. Positive group, amusement − tenderness +0.08 [−0.19, 0.34], p 0.54 | no evidence of separation |
| H6 | Raw POPANE files not present on 2026-11-16. Run 2026-11-23 as set here, once the files were in (Part 19): r +0.14 [−0.01, +0.28], p 0.074, 42 of 50 | not supported |

- **All four checks give the same verdicts:** the 0.02 µS threshold, the binary filter, the run
  without P68, and the rank tests next to every test.
- **The pulse rate (R) agrees with the state on every test.** H2 +0.37, p 0.014; H3 (b) +0.45,
  p 0.039; mixed-group D −0.01. So the state carries what the raw pulse timing shows, and is not
  an artefact of the estimator.
- **Pulse size (Z) separates nothing.** No group or valence contrast is significant. Size is
  slightly up in films overall (+0.07 [0.00, 0.14], p 0.04), the same in every group.
- **What drives H2 is the neutral group falling.** The neutral films bring arousal below baseline
  (−0.31), while the emotional films hold it near baseline (−0.10 to +0.19, none significant). So
  "emotional films raise it more than neutral ones" holds as a difference between groups, but no
  film group rises significantly above its own baseline. Responding fades over a session (Next
  steps, item 3), so baseline-to-film is not a clean "no film" contrast.
- **H3 against H4, within person.** In the mixed group the same people show no valence difference,
  which is the H3 side of the question the proposal set the mixed group to settle. H3 as a whole
  fails because the negative group does not rise above the neutral group (n = 10, wide CI). The
  negative-group and positive-group comparison is not shown equivalent either. So the
  between-group evidence is weak in both directions.
- **H5 is borderline for the negative films** (p 0.062; R p 0.056), on 10 people, with fear,
  disgust and sadness each a single film.
- **One outlier.** One neutral-group participant falls by about 2 logit units. The rank tests,
  which it cannot dominate, give the same verdicts.

## Part 15: the tonic level as a third observation (variant `tonic14`, exploratory)

Set 2026-11-23, after Part 14's results, at Piyush's request. **This is a new, named variant. It
cannot change any H1–H6 verdict in Part 14.** The question: does adding the slow skin-conductance
level change the hypothesis results?

**The problem it has to handle.** In Data Challenge 2, a continuous feature observed in every bin
took over the state (Problem 4C, cortisol; Problem 5C, sleep). Each sample was weighted by
1/(noise variance), and there were far more samples than pulses. The tonic level would be the
extreme case:
- it is in every bin (about 1,200 per person, against about 190 pulses);
- it is nearly noise-free (a spline);
- its seconds are not independent (knots every 14 s, the frozen stage-1 setting).

**The weighting rule, option 3** (chosen 2026-11-23, before any tonic state was computed): the
filter sees the tonic level once per 14 s, the stage-1 knot spacing, which is about one independent
value per knot interval. It uses bins 7, 21, 35, …, and the other bins have no tonic observation.
There is no variance floor or balance rule (the Data Challenge's lines 174 and 165), so this rule
is the only control on the tonic level's weight.

**The variant.**
- **Feature:** `tonic` from `P<id>_own4_signals.csv` (µS), averaged over each 1 s bin. Not
  detrended.
- **Model:** the main filter unchanged (b₀ pinned, b₁ = 1, EM for v_e, r₀, r₁, v_r, stop on
  r₁ ≤ 0), plus s_k = s₀ + s₁·x_k + N(0, v_s) in the observed bins. The continuous terms are those
  of `filter_one_mpp_one_cont.m`, with b₁ = 1. No sign constraint on s₁: the pulses fix the sign of
  x.
- **Starting values:** s₀ = mean, s₁ = SD and v_s = variance/2 of the observed tonic values (the
  Problem 3F rule, as for the sizes).
- **Which filter:** MPP + tonic when it converges. Otherwise binary + tonic (the same filter
  without sizes). If neither converges, the participant keeps the main analysis's chosen state, and
  this is counted.
- **Regression check:** the same code, with sizes off, tonic in every bin and the Data Challenge's
  starting values, must reproduce `filter_one_bin_one_cont` on Problem 3C exactly.

**Synthetic check** (`s2_synth_tonic`), set before it runs. It uses Part 13's generator at
r₁ = 0.5, Δ ∈ {−0.5, 0, 0.5, 1.0}, and the 50 real layouts, giving 200 sessions per scenario,
plus a true tonic level in µS:
- **A, informative:** s = 5 + 1·x + N(0, 0.05²) in every second.
- **B, drift:** s = the layout participant's own real tonic level + 1·x. The real tonic level's own
  course, including its fall over the session, is drift unrelated to the true state. This is the
  habituation confound.

| | Criterion (chosen filter) | Pass |
|---|---|---|
| T1 | Contrast recovery, per scenario | r(Δ̂, Δ_true) ≥ 0.8 and slope 0.7–1.3 |
| T2 | Bias per Δ level, per scenario | \|mean(Δ̂ − Δ_true)\| ≤ 0.1 |
| T3 | Baseline-referenced coverage (C3′) | median 0.85–0.99 |
| T4 | Converged (MPP + tonic or binary + tonic) | ≥ 95% |

Also reported: the same sessions through the main filter (no tonic), as the reference.
- **If A passes and B fails T1 or T2,** the tonic level leaks drift into the state. The real run is
  still made, as requested, but its results are reported as confounded, not as evidence.
- **If A fails,** the variant does not work and the real run is not made.

**Real run:** `s2_run_tonic` writes `data/stage2_params_tonic.csv` and
`data/stage2_states_005_tonic.csv`. Then `s2_tests(pwd, 'variant', 'tonic')` writes
`data/stage2_tests_tonic.csv` and `results/figures/stage2_hypotheses_tonic.png`. It runs the main
check only (0.05 µS) and leaves Part 14's files untouched.

**Regression check: pass.** `s2_filter_tonic` with sizes off reproduces `filter_one_bin_one_cont`
on Problem 3C exactly (difference 0). The five earlier `s2_test_dc` cases are unchanged.

**Synthetic check result** (`data/stage2_synth_tonic_scores.csv`, 2026-11-23). The main filter,
run on the same sessions, is the reference.

| | A, informative | B, drift | Main filter (both) |
|---|---|---|---|
| T1 r, slope | 0.972, 1.25 (pass) | 0.809, 0.98 (pass) | 0.967, 1.00 |
| T2 bias by Δ (−0.5, 0, 0.5, 1.0) | −0.18, −0.01, +0.18, +0.16 (**fail**) | +0.23, +0.28, +0.30, +0.15 (**fail**) | −0.01 to +0.04 |
| T3 C3′ coverage median | 0.68 (**fail**) | 0.26 (**fail**) | 0.98 |
| T4 converged | 0.97 (pass) | 0.95 (pass) | |

- **A: the tonic level still dominates at one observation per 14 s.** The fitted tonic noise is
  tiny. The state follows the tonic level closely (correlation with the true state 0.985), but its
  scale is wrong: s₁ comes out at 0.81 against a true 1, so x is stretched by about 1.25, and the
  film effect is overstated by up to 0.18. The 95% bands are far too narrow (coverage 0.68): the
  filter trusts the tonic level as near-exact.
- **B: the real tonic level's own course leaks into the state.** It adds +0.15 to +0.30 to the
  film effect. This is the habituation confound named in the plan. The state agrees with the true
  state less well than the main filter's does (correlation 0.76 against about 0.97).
- **One real participant was looked at before the check, as a smoke test:** on P99, the fitted
  tonic noise was 0.0007 µS², and the state correlated 0.45 with the main state. No contrast was
  computed.
- **Decision, by the rule above: A fails, so the variant does not work, and the real run is not
  made.** Option 3 alone does not control the tonic level's weight. Any fix would be a new variant
  with a new name, tested once, and the setting must not be refined until it passes (rule 5):
  - a variance floor (the Data Challenge's line 174);
  - a separate drift state (option 5).
- **Override (2026-11-23, Piyush): the real run is made anyway.** This was decided after the
  synthetic check failed, and against the rule above. Its H1–H6 results come from a variant that
  failed validation: they are reported for completeness, are not evidence, and change no verdict.

**Real run** (`data/stage2_params_tonic.csv`, `data/stage2_states_005_tonic.csv`, 2026-11-23).
- **Which filter was used:**
  - MPP + tonic: 35 participants;
  - binary + tonic: 13 (MPP + tonic stopped on r₁ ≤ 0, against 6 in the main analysis);
  - main state kept: 2, P21 and P168 (neither tonic filter converged).
- **Fitted tonic noise:** SD about 0.02 µS for most participants, so the filter treats the tonic
  level as near-exact, as in synthetic scenario A.
- **The tonic state is mostly not the main state:** R² against it is often below 0.2.
- **Negative tonic slope (s₁ < 0) in 4 participants** (P10, P20, P152, P179): their tonic level
  falls as the pulse-defined arousal rises.

**H1–H6 on the variant** (`s2_tests(pwd, 'variant', 'tonic')`: `data/stage2_tests_tonic.csv`,
`data/stage2_block_summary_tonic.csv`, `results/figures/stage2_hypotheses_tonic.png`). State (S),
change from own baseline:

| | Main analysis (Part 14) | Tonic variant |
|---|---|---|
| H1, all 50 | −0.08, p 0.30, not supported | +0.30 [0.06, 0.54], p 0.014, "supported" |
| H2, emotional − neutral | +0.37, p 0.023, supported | +0.24, p 0.35, not supported |
| H3 (a) negative − neutral | +0.20, p 0.31 | +0.14, p 0.69 |
| H3 (b) positive − neutral | +0.50, p 0.023 | +0.13, p 0.69 |
| H3 (c) mixed group D, 90% CI | +0.05 [−0.30, 0.41], equivalent | +0.13 [−0.05, 0.31], equivalent |
| H4 | not supported | not supported |
| H5 | no evidence of separation | no evidence of separation |

- **H1 "flips" by about +0.38,** the size of the drift bias found in synthetic scenario B (+0.15
  to +0.30, plus a stretched scale). It is not a finding: the tonic level is higher during films
  than at baseline for most people, and the variant turns that into "arousal".
- **The variant's state no longer agrees with the pulses.** The neutral group is +0.15 in the
  tonic state, while its pulse rate falls (R −0.25, unchanged). By Part 14's rule, a state that
  disagrees with the raw pulse statistics says something about the estimator: here, that it
  reports the tonic level, not the bursts.
- **The results that do not depend on the tonic level hold:** within the mixed group, negative
  and positive films are equivalent; H4 is not supported; and there is no separation within a
  valence.
- **Conclusion for the report:** adding the tonic level as a continuous observation, weighted
  only by observing it once per 14 s, makes the state follow the slow level and its session
  drift. It failed validation, and on real data it overturns H1 and H2 in the direction the
  synthetic check predicted. Leaving the tonic level out (Part 13) is the right choice for this
  model. A version that could work needs a separate drift state (option 5) or an explicit weight,
  validated before use.

## Part 16: tonic14 with a balanced floor (variant `tonic14b`, exploratory)

Set 2026-11-23, after Part 15's results, at Piyush's request, and written down before any
`tonic14b` state was computed. **A new, named variant; it cannot change any H1–H6 verdict in
Part 14.** It changes one thing from `tonic14`: a cap on the tonic level's weight.

**Why.** Part 15 found two problems:
- **Overconfidence (scenario A).** The fitted tonic noise collapsed (SD about 0.02 µS). One tonic
  observation then carried several hundred times the information about the state that the pulses
  carry over the same 14 s. The state became a copy of the tonic level, with bands far too narrow.
- **Drift (scenario B).** The tonic level's own slow course entered the state.

The floor targets the first problem directly. For the second it can only reduce the leak, not
remove it.

**The rule (balanced floor).** A tonic observation may carry at most as much information about the
state as the pulses are expected to carry between two tonic observations:

  s₁² / v_s ≤ I_pulse = g · p̄(1 − p̄)

- g = 14, the spacing of the tonic observations;
- p̄ is the participant's pulse fraction per bin, the same one that pins b₀;
- p̄(1 − p̄) is the information about x_k in one bin's pulse/no-pulse observation (the filter's own
  term). The size observations are not counted, so the cap is the same for MPP + tonic and
  binary + tonic.

It is applied as v_s ← max(v_s, s₁² / I_pulse), to the starting value and after every M step. It
caps the ratio, not v_s alone, so EM cannot restore the weight by inflating s₁. The stop on a
fitted v_s ≤ 0 is checked before the floor, as in `tonic14`. The rule comes from the filter's own
terms, not from any real result.

**Everything else is `tonic14` unchanged:**
- the tonic feature, observed at bins 7, 21, 35, …;
- the model and the starting values (apart from the floor);
- the filter choice (MPP + tonic, else binary + tonic, else the main state);
- the regression check: with the floor off, the code must still reproduce Problem 3C exactly, and
  Part 15's tonic14 state for P99.

**Synthetic check:** Part 15's sessions (same seeds), scenarios A and B, and the same criteria
T1–T4 (Part 15's table). Also reported: how often the floor is active at the end.

**Decision rule, overridden in advance (2026-11-23, Piyush):**
- the real run and H1–H6 are made **whatever the synthetic check shows**;
- if A or B fails any of T1–T4, the variant's H1–H6 results are reported as coming from a variant
  that failed validation: for completeness, not as evidence;
- whatever the result, the floor is not re-tuned (rule 5).

**Expected, written before the run:**
- A: the bands widen and the scale bias shrinks, so T2 and T3 may pass;
- B: the drift bias shrinks but does not vanish;
- H1 remains confounded with time on task in any case, because the baseline always comes first.

**Commands:**
- synthetic check: `s2_synth_tonic(pwd, 'tonic14b')` → `data/stage2_synth_tonic14b_scores.csv`;
- real run: `s2_run_tonic(pwd, 'tonic14b')` → `data/stage2_params_tonic14b.csv`,
  `data/stage2_states_005_tonic14b.csv`;
- tests: `s2_tests(pwd, 'variant', 'tonic14b')` → `data/stage2_tests_tonic14b.csv`,
  `data/stage2_block_summary_tonic14b.csv`, `results/figures/stage2_hypotheses_tonic14b.png`.

Part 14's and Part 15's files are not touched.

**Regression check: pass.** With the floor off:
- the five `s2_test_dc` cases and Problem 3C are identical (difference 0);
- P99's tonic14 state matches `data/stage2_states_005_tonic.csv` to 5·10⁻¹⁶.

A smoke test on P99, with the floor on: in tonic14, one tonic observation had carried 1,083 times
I_pulse. Now the cap holds it at exactly 1, and the tonic noise SD is 0.70 µS instead of about
0.03.

**Synthetic check result** (`data/stage2_synth_tonic14b_scores.csv`, 2026-11-23). The same
sessions as Part 15; tonic14 is shown for comparison.

| | A, tonic14b | B, tonic14b | A, tonic14 | B, tonic14 | Main filter |
|---|---|---|---|---|---|
| T1 r, slope | 0.958, 1.03 (pass) | 0.943, 0.91 (pass) | 0.972, 1.25 | 0.809, 0.98 | 0.967, 1.00 |
| T2 bias by Δ (−0.5, 0, 0.5, 1.0) | +0.00, −0.01, +0.07, +0.03 (pass) | +0.15, +0.07, +0.03, −0.01 (**fail**) | −0.18 to +0.18 | +0.15 to +0.30 | −0.01 to +0.04 |
| T3 C3′ coverage median | 0.906 (pass) | 0.777 (**fail**) | 0.68 | 0.26 | 0.98 |
| T4 converged | 1.000 (pass) | 0.995 (pass) | 0.97 | 0.95 | |

- **The floor was active at the end in every converged session, A and B.** Left alone, EM always
  gives the tonic level more weight than the pulses.
- **A passes all four criteria: the overconfidence is fixed.** The stretched scale is gone (slope
  1.03 against 1.25), the bias is within 0.07, and the bands are honest (coverage 0.91).
- **B still fails, but the leak is much smaller:**
  - the drift bias is at most +0.15 (at Δ = −0.5) and within 0.1 at the other levels, against
    +0.15 to +0.30 in tonic14;
  - coverage is 0.78, against 0.26;
  - the state's correlation with the truth is a median of 0.85, against 0.76.

  The real tonic level's own course still enters the state, as predicted: the floor can reduce
  the drift, but not remove it.
- **Warnings:** in some sessions the 2 × 2 regression in the M step (sizes, line 120; tonic, line
  130) was near-singular during EM, which happens when the state is close to flat. Every
  converged state was finite, and T4 passed.
- **Decision:** B fails T2 and T3, so the variant fails validation. By the rule set in advance,
  the real run and H1–H6 are made anyway (Piyush), and are reported as coming from a variant
  that failed validation. The floor is not re-tuned.

**Real run** (`data/stage2_params_tonic14b.csv`, `data/stage2_states_005_tonic14b.csv`,
2026-11-23).
- **Which filter was used:** MPP + tonic for 43 participants and binary + tonic for 7 (P15, P49,
  P106, P111, P115, P118, P178). None kept the main state. The main analysis had 44 MPP and 6
  binary.
- **The floor is active in 48 of 50.** The other two had already settled at 0.03 and 0.18 times
  the cap.
- **The state now mostly agrees with the main state:** R² against it has a median of 0.65, with 5
  participants below 0.2. In tonic14 the median was 0.25, with 23 below 0.2.
- **s₁ < 0** in the same 4 participants as tonic14 (P10, P20, P152, P179).
- **s₁ is large in a few participants** (P15 +9.5, P20 −9.2, P111 +5.9, P115 +5.0, P76 +4.9), each
  with a matching large tonic noise. The cap fixes the ratio, so a stretched s₁ just means a
  compressed state scale for that person.
- **P111's MPP + tonic run** gave near-singular warnings before it stopped on r₁ ≤ 0. Its
  binary + tonic state, the one used, converged.

**H1–H6 on the variant** (`s2_tests(pwd, 'variant', 'tonic14b')`:
`data/stage2_tests_tonic14b.csv`, `data/stage2_block_summary_tonic14b.csv`,
`results/figures/stage2_hypotheses_tonic14b.png`). State (S), change from own baseline:

| | Main analysis (Part 14) | tonic14 (Part 15) | tonic14b |
|---|---|---|---|
| H1, all 50 | −0.08, p 0.30, not supported | +0.30, p 0.014, "supported" | +0.10 [−0.09, 0.28], p 0.29, not supported |
| H2, emotional − neutral | +0.37 [0.05, 0.69], p 0.023, supported | +0.24, p 0.35 | +0.34 [−0.06, 0.73], p 0.091, not supported |
| H3 (a) negative − neutral | +0.20, p 0.31 | +0.14, p 0.69 | +0.12, p 0.62 |
| H3 (b) positive − neutral | +0.50, p 0.023 | +0.13, p 0.69 | +0.45 [−0.03, 0.94], p 0.064 |
| H3 (c) mixed group D, 90% CI | +0.05 [−0.30, 0.41], equivalent | +0.13 [−0.05, 0.31], equivalent | +0.08 [−0.13, 0.29], equivalent |
| H4 | not supported | not supported | not supported |
| H5 | no evidence of separation | no evidence of separation | no evidence of separation (negative group p 0.15) |
| Group means: neutral, negative, positive, mixed | −0.31, −0.10, +0.19, +0.11 | +0.16, +0.30, +0.29, +0.61 | −0.10, +0.02, +0.35, +0.32 |

- **The estimates are close to the main analysis again.** H2 is +0.34 against +0.37, and
  H3 (b) +0.45 against +0.50. tonic14 had pulled both towards zero.
  - The intervals are wider, so H2 drops just above 0.05 (p 0.091).
  - The verdict changes, but the size of the effect does not. Its 90% interval still excludes
    zero ([+0.01, +0.66]).
- **H1 moves by about +0.18** relative to the main analysis (tonic14: +0.38). That is in the
  range of the drift bias left in scenario B (up to +0.15), and it stays not supported.
- **The state agrees with the pulses again.** The neutral group is −0.10, the same sign as its
  pulse rate (R −0.25); in tonic14 it was +0.16. The negative group is close to zero in both S and
  R.
- **The within-person results hold, as in every analysis:**
  - within the mixed group, negative and positive films are equivalent;
  - H4 is not supported;
  - there is no separation within a valence.
- **P178 is the largest outlier** (−2.4), as in the main analysis (−2.1).
- **Conclusion for the report:**
  - **The balanced floor fixes the overconfidence.** It passes scenario A, which tonic14 failed.
  - **It shrinks the drift leak by half or more, but does not remove it.** On real data the H1
    shift falls from +0.38 to +0.18; in scenario B the mean bias falls from +0.24 to +0.06. B fails, so
    the variant is not validated. Its H1–H6 results are not evidence and change no verdict.
  - **The results agree with the main analysis in direction and size, not in significance.**
  - **The tonic level adds no hypothesis evidence beyond the pulses.** Its remaining influence is
    partly drift. A version that could be validated needs the drift handled in the model (a
    separate drift state, option 5) or a drift-free input (the tonic level's changes, not its
    level).

## Part 17: the differenced tonic level (variant `tonic14d`, exploratory)

Set 2026-11-23, after Part 16's results, at Piyush's request, and written down before any
`tonic14d` state was computed. **A new, named variant; it cannot change any H1–H6 verdict in
Part 14.** It changes one thing from `tonic14b`: the filter sees the tonic level's changes, not
its level.

**Why.** Part 16's floor fixed the overconfidence. What was left is the tonic level's own slow
course (drift), which a weight can only shrink. Differencing removes a constant offset entirely,
turns a linear drift into a constant (absorbed by s₀), and leaves a curved drift only as a small,
slowly varying offset.

**The input.**
- The tonic level is read at bins 7, 21, 35, … (as in tonic14).
- The filter sees each change, Δs_j = s(7 + 14j) − s(7 + 14(j − 1)), placed at the midpoint bin
  14j (bins 14, 28, 42, …). Every other bin has no continuous observation.
- That gives one observation per 14 s, as before.

**What the model then says.** The filter is unchanged: Δs_k = s₀ + s₁·x_k + N(0, v_s). With the
change as input, this means the tonic level *rises at a rate set by arousal*, so the level
integrates arousal. Physiologically this is the leaky-integrator view of sweat accumulation, in
contrast to tonic14's assumption that the level is proportional to arousal. Everything else is
`tonic14b` unchanged:
- the balanced floor, s₁²/v_s ≤ 14·p̄(1 − p̄);
- the starting values, computed from the observed differences;
- the filter choice (MPP + tonic, else binary + tonic, else the main state).

**Synthetic check.** Part 15's sessions (same seeds; r₁ = 0.5, Δ ∈ {−0.5, 0, 0.5, 1.0}, the 50
layouts), with three true tonic levels:
- **A, level ∝ arousal** (Parts 15–16's A): s = 5 + x + N(0, 0.05²). The differenced model is
  misspecified here. Reported, not a criterion.
- **Ai, integrator:** s_k = 5 + γ·Σ_{j≤k} x_j + N(0, 0.05²), with γ = 0.01 µS per second per logit.
  For example, a clip held at +0.5 raises the level by 0.6 µS over 120 s. γ was chosen a priori as
  a plausible size, not fitted to the real data.
- **Bi, drift:** s = the layout participant's own real tonic level + γ·Σ_{j≤k} x_j. Its own course
  is drift unrelated to the true state, as in Part 15's B.

Criteria T1–T4 (Part 15's table), on Ai and Bi.

**Decision rule.**
- **Pass** if Ai and Bi both pass T1–T4. Then `s2_run_tonic(pwd, 'tonic14d')` and
  `s2_tests(pwd, 'variant', 'tonic14d')` run on the 50.
- **Fail** otherwise. The real run is not made unless Piyush decides otherwise, and γ and the input
  are not re-tuned (rule 5).

**Outputs:**
- `data/stage2_synth_tonic14d_scores.csv`;
- and if it passes: `data/stage2_params_tonic14d.csv`, `data/stage2_states_005_tonic14d.csv`,
  `data/stage2_tests_tonic14d.csv`, `data/stage2_block_summary_tonic14d.csv` and
  `results/figures/stage2_hypotheses_tonic14d.png`.

**Regression check: pass.** The tonic14 and tonic14b inputs now come from `s2_tonic_input`
(new). Rebuilt that way, P99's tonic14 and tonic14b states match the saved ones to 5·10⁻¹⁶.
- **Smoke test, one synthetic Ai session** (no real participant was looked at): the fit
  converged, with s₁ 0.11 against a true 14γ = 0.14, and the tonic-change noise SD 0.072 against
  an expected 0.071 (√2 · 0.05).

**Synthetic check result** (`data/stage2_synth_tonic14d_scores.csv`, 2026-11-23):

| | A, level ∝ arousal (not a criterion) | Ai, integrator | Bi, real drift + integrator | Main filter |
|---|---|---|---|---|
| T1 r, slope | 0.968, 1.00 | 0.980, 1.14 (pass) | 0.973, 1.06 (pass) | 0.967, 1.00 |
| T2 bias by Δ (−0.5, 0, 0.5, 1.0) | −0.00, +0.02, +0.04, −0.01 | −0.13, −0.00, +0.11, +0.09 (**fail**) | −0.07, −0.01, +0.05, +0.02 (pass) | −0.01 to +0.04 |
| T3 C3′ coverage median | 0.98 | 0.98 (pass) | 0.97 (pass) | 0.98 |
| T4 converged | 1.00 | 1.00 (pass) | 1.00 (pass) | |
| Floor active | 0% | 22% | 34% | |
| s₁ median | −0.00 | 0.08 (true 0.14) | 0.10 (true 0.14) | |

- **The drift problem is solved.** Bi, the real tonic course plus the integrator, passes all four
  criteria. Bias is within 0.07 and the bands are honest. In Parts 15–16, the matching scenario B
  failed with drift biases up to +0.30.
- **The clean integrator case, Ai, fails T2 narrowly.** The scale is stretched (slope 1.14), so
  the film effect is overstated by 0.11–0.13 at Δ = ±0.5, against a limit of 0.1. s₁ comes out at
  about 0.08 against a true 0.14, which stretches x. The bands stay honest (coverage 0.98).
- **It fails safe when the tonic level carries no rate information.** In A (level ∝ arousal), s₁
  comes out at about 0. The tonic changes are ignored, and the state is the main state
  (R² 0.999).
- **Warnings:** near-singular M-step regressions in some sessions, as in Part 16. Every converged
  state was finite.
- **Decision, by the rule above: Ai fails T2, so the variant fails validation, and the real run is
  not made.** The failure is narrow: it is a scale stretch, not drift. γ and the input are not
  re-tuned (rule 5). This is the first tonic variant whose drift scenario passes.

## Part 18: the tonic link calibrated from the pulses (variant `tonic14c`, exploratory)

Set 2026-11-23, after Part 17's results, at Piyush's request, and written down before any
`tonic14c` state was computed. **A new, named variant; it cannot change any H1–H6 verdict in
Part 14.** It changes one thing from `tonic14d`: how the tonic link is estimated.

**Diagnosis, from Part 17's synthetic scores only (no real data looked at).** The stretch sits in
the sessions where the floor was not active:
- Ai, floor not active (157 sessions): s₁ 0.069 against a true 0.14, bias +0.12 at Δ = +0.5 and
  +1.0;
- Ai, floor active (43 sessions): s₁ 0.110, bias +0.06 and +0.04.

When EM estimates the state and the link together, it can lower s₁ and enlarge x while fitting the
tonic changes equally well. The pulses fix x's scale only weakly (b₁ = 1), so EM settles too low on
s₁ and the state is stretched.

**The calibration (regression calibration), per participant:**
1. Take the main analysis's chosen state (pulses only) at every bin: x̂_k and its variance v_k.
   Their scale comes only from the pulses.
2. At the bins where a tonic change is observed (14, 28, …), fit Δs_j = s₀ + s₁·x̂_j by least
   squares.
3. Set v_s = (mean squared residual) − s₁²·mean(v_j). The second term removes the part of the
   spread that is only the pulse state's own uncertainty. If this leaves v_s ≤ 0, it is set to
   10⁻¹² and the floor takes over.
4. Apply the balanced floor (Part 16): v_s ≥ s₁² / (14·p̄(1 − p̄)).
5. Run the tonic filter with s₀, s₁ and v_s **held fixed**. EM still estimates v_e, r₀, r₁ and v_r.

Because the slope is fitted against the pulses' best guess of x, not against a noisy measurement of
it, it is not shrunk. The tonic changes can then sharpen the state but cannot rescale it.

**Everything else is `tonic14d` unchanged:**
- the input: tonic changes at bins 14, 28, …;
- the floor;
- the filter choice: MPP + tonic, else binary + tonic, else the main state. On real data the
  calibration uses the main analysis's chosen state from `data/stage2_states_005.csv`, with its
  variance rebuilt from its 95% band.

**Synthetic check on fresh seeds.** Part 17's sessions were used to diagnose the failure and design
this fix, so they no longer test it fairly.
- **The decision uses fresh sessions:** the same generator, scenarios (A, Ai, Bi), γ = 0.01, 50
  layouts and Δ levels, with seeds `900000 + 1000·cond + i` instead of `1000·cond + i`. No earlier
  stage-2 run used them.
- **For comparison only:** the old seeds are also run. They do not enter the decision.

**Criteria:** T1–T4 (Part 15's table) on Ai and Bi, fresh seeds.

**Decision rule:**
- **Pass** if Ai and Bi both pass T1–T4 on the fresh seeds. Then `s2_run_tonic(pwd, 'tonic14c')`
  and `s2_tests(pwd, 'variant', 'tonic14c')` run on the 50, with no override needed.
- **Fail** otherwise. The real run is not made unless Piyush decides otherwise, and no further
  refinement of the tonic link follows (rule 5).

**Outputs:**
- `data/stage2_synth_tonic14c_scores.csv` (fresh seeds);
- `data/stage2_synth_tonic14c_oldseeds_scores.csv` (comparison);
- if it passes: `data/stage2_params_tonic14c.csv`, `data/stage2_states_005_tonic14c.csv`,
  `data/stage2_tests_tonic14c.csv`, `data/stage2_block_summary_tonic14c.csv` and
  `results/figures/stage2_hypotheses_tonic14c.png`.

**Regression check: pass.**
- The six `s2_test_dc` cases are identical (difference 0).
- P99's tonic14 and tonic14b states match the saved ones to 5·10⁻¹⁶.
- A tonic14d synthetic session (Ai, Δ = −0.5, first layout) reproduces its saved Δ̂ exactly.
- Smoke test on that old-seed session: the calibrated link gave s₁ 0.105 and a noise SD of 0.072
  µS, and the fit converged in 10 iterations.

**Synthetic check result** (2026-11-23). The decision uses the fresh seeds
(`data/stage2_synth_tonic14c_scores.csv`). The old seeds
(`data/stage2_synth_tonic14c_oldseeds_scores.csv`) are for comparison only.

| | Ai, fresh | Bi, fresh | A, fresh (not a criterion) | Ai, old seeds | Bi, old seeds | Main filter, fresh |
|---|---|---|---|---|---|---|
| T1 r, slope | 0.978, 1.06 (pass) | 0.973, 1.05 (pass) | 0.970, 1.00 | 0.976, 1.05 | 0.973, 1.04 | 0.969, 1.01 |
| T2 bias by Δ (−0.5, 0, 0.5, 1.0) | −0.03, −0.03, +0.01, +0.07 (pass) | −0.02, −0.03, −0.01, +0.07 (pass) | +0.04, −0.02, −0.02, +0.07 | −0.06, +0.00, +0.07, +0.02 | −0.05, −0.01, +0.04, +0.01 | +0.03, −0.02, −0.02, +0.07 |
| T3 C3′ coverage median | 0.989 (pass) | 0.978 (pass) | 0.984 | 0.989 | 0.979 | 0.980 |
| T4 converged | 1.00 (pass) | 1.00 (pass) | 1.00 | 1.00 | 1.00 | |
| Floor active | 57% | 16% | 0% | 52% | 21% | |
| s₁ median (true 0.14) | 0.10 | 0.10 | −0.00 | 0.10 | 0.10 | |

- **Pass on the fresh seeds:** Ai and Bi meet all four criteria. By the rule above, the real run is
  made with no override.
- **The stretch is gone.** Ai's slope is 1.06 (tonic14d: 1.14), and its worst bias is 0.07
  (tonic14d: 0.13).
  - **The fresh seeds' +0.07 at Δ = 1.0 is in the main filter too** (+0.07), so it belongs to
    those draws, not to the tonic level.
- **The drift stays handled** (Bi). When the tonic changes carry no information (A), the variant
  falls back to the main state (R² 0.998).
- **The old seeds agree.** The fix passes on the sessions it was designed on and on fresh ones
  alike.
- **The calibrated s₁ is still below the truth** (0.10 against 0.14). Because the scale is fixed
  by the pulses, this only lowers the tonic level's weight, the safe direction, and causes no
  stretch.
- **Warnings:** near-singular M-step regressions in some synthetic sessions, as in Parts 16–17.
  Every converged state was finite.

**Real run** (`data/stage2_params_tonic14c.csv`, `data/stage2_states_005_tonic14c.csv`,
2026-11-23). No warnings.
- **Which filter was used:** MPP + tonic for 48 participants, binary + tonic for 2 (P15 and P178,
  MPP stopped on r₁ ≤ 0), and the main state for none.
  - The main analysis had 6 binary. P106, P111, P115 and P168 now converge with MPP + tonic.
- **The link:** calibrated s₁ has a median of 0.09 (range −0.0001 to 0.35), and the tonic-change
  noise SD a median of 0.10 µS.
  - No participant has a clearly negative link. tonic14 and tonic14b had 4.
  - The floor was active for 18 of 50.
- **The state stays close to the main state:** R² median 0.94, with none below 0.2.

**H1–H6 on the variant** (`s2_tests(pwd, 'variant', 'tonic14c')`:
`data/stage2_tests_tonic14c.csv`, `data/stage2_block_summary_tonic14c.csv`,
`results/figures/stage2_hypotheses_tonic14c.png`). State (S), change from own baseline:

| | Main analysis (Part 14) | tonic14c |
|---|---|---|
| H1, all 50 | −0.08 [−0.24, 0.07], p 0.30, not supported | −0.06 [−0.20, 0.08], p 0.38, not supported |
| H2, emotional − neutral | +0.37 [0.05, 0.69], p 0.023, supported | +0.37 [0.09, 0.64], p 0.010, supported |
| H3 (a) negative − neutral | +0.20, p 0.31 | +0.18 [−0.20, 0.56], p 0.34 |
| H3 (b) positive − neutral | +0.50 [0.08, 0.93], p 0.023 | +0.50 [0.10, 0.89], p 0.016 |
| H3 (c) mixed group D, 90% CI | +0.05 [−0.30, 0.41], equivalent | +0.06 [−0.26, 0.38], equivalent |
| H3 verdict | not supported | not supported |
| H4 | not supported | not supported |
| H5 | no evidence of separation | no evidence of separation (negative group p 0.24) |
| Group means: neutral, negative, positive, mixed | −0.31, −0.10, +0.19, +0.11 | −0.28, −0.10, +0.22, +0.14 |

- **Every verdict is the same as in the main analysis,** and the estimates agree to within 0.03.
- **The intervals are slightly narrower** (H2: [0.09, 0.64] against [0.05, 0.69]). The tonic
  changes add a little information, which is what a validated extra observation should do.
- **The state agrees with the pulses:** the neutral group is −0.28, against a pulse rate of R −0.25.
- **P178 is still the largest outlier** (−1.7; main −2.1).
- **Conclusion for the report:**
  - **This is the first tonic variant to pass validation, and the first on fresh seeds.** It uses
    the tonic level's 14 s changes, with the link calibrated from the pulse-only state and the
    balanced floor.
  - **It confirms the main analysis.** The verdicts are the same and the estimates within 0.03,
    with slightly tighter intervals.
  - **It is still not a replacement** for the Part 14 analysis, which was set in advance. It is an
    exploratory variant reached after three failed ones. It is reported as a robustness check:
    the pulse-only results hold when the tonic level is added in a validated way.
  - **The path is itself a result.** The raw tonic level fails on overconfidence and on drift. A
    weight cap fixes the overconfidence. Differencing fixes the drift. Calibrating the link from
    the pulses fixes the scale.

## Part 19: the affect slider (H6, and the evaluation of Oliver et al. 2025)

Set 2026-11-23, written before the slider was compared with any state. The professor pointed us to
Oliver, Zhang, Raju, Murrough & Faghih (2025), *PLOS Mental Health* 2:e0000463. That paper:
- estimates arousal from skin-conductance pulses with an MPP filter, as we do;
- uses self-reported arousal only to **evaluate** the state, never as a model input;
- reports two analyses:
  1. **High/low agreement.** HAI (High Arousal Index) = P(state > the person's median state). The
     person's top HAI periods are compared with the highest-rated stimuli, with a per-person rating
     cutoff tuned in-sample. Accuracy 69%, sensitivity 50%, specificity 73%.
  2. **Direction of change.** For consecutive stimuli in different rating groups, does the state go
     up when the rating does? Accuracy 54.5–63.6% across the cutoffs swept, against 50% chance.
     Their state–rating R² was 0.03.

**Data** (`s2_slider` → `data/stage2_slider_1hz.csv`). The raw files come from Piyush's copy of
the POPANE Study 6 archive (`study6.zip`, files `<id>_All.csv`). The 50 analysed participants were
extracted to `data/raw/Study6/S6_P<id>_All.csv` (local only, not shared).
- **Same recordings as stage 1:** raw EDA matches stage 1's input `y` to within 0.0025 µS on the
  same clock (P99).
- **The slider on the stage-2 bins:** its mean in each 1 s bin, and whether it moved in that bin.
  No bin is missing.
- **Cross-check against Eleftherios's file** (`slider_study6_all.csv`, shared 2026-11-23, not in
  the repository: 113 MB):
  - it holds the slider at 10 Hz on the recording clock, with marker, group and selected flag, for
    all 187 Study 6 participants;
  - its `selected` flag picks exactly our 50;
  - binned the same way, it matches `data/stage2_slider_1hz.csv` (from the 1000 Hz raw files):
    median difference 0.000, 99th percentile 0.018, worst 0.28. The worst bins (P115, P175) are
    where the slider moves fast, so 10 Hz and 1000 Hz averages differ;
  - the analysis uses the 1 Hz table built from the raw files. The 1 Hz table is in the
    repository (2 MB), so the analysis can be re-run without the raw files.

**What the slider looks like** (the slider alone, before any comparison with a state):
- **The recorded scale is 0–10, not 1–10:** values run from 0.0 to 9.9. The resting position, the
  baseline median, is 4.9 (IQR 4.8–5.0).
- **At baseline the slider mostly stays still:** in the median participant it never moves.
- **During the films its range has a median of 3.3.** Nine participants move it less than 0.5
  across all six films: P12, P26, P34, P111, P113, P138, P159, P168 and P179.
- **Clip means** (valence; intensity measured from 5.5):

  | Emotion | Valence | Intensity |
  |---|---|---|
  | anger | 4.1 | 1.7 |
  | fear | 4.4 | 1.6 |
  | disgust | 4.3 | 1.4 |
  | amusement | 6.0 | 1.1 |
  | tenderness | 5.4 | 1.0 |
  | neutral | 5.2 | 1.0 |
  | sadness | 4.7 | 0.9 |

### A. H6, as set in Part 14

- **Run once, as written:** per person, the Spearman r across the 6 clips between the mean state
  and the mean |a − 5.5|; then Fisher z and a one-sample t.
- **The reader is fixed, not the test.** Part 14's reader used automatic header detection, which
  fails on the real files' `#` header. It now finds the header line itself.
- **Re-running `s2_tests(pwd)` rewrites Part 14's files,** so they are backed up first. Every H1–H5
  row must come out identical, and only the H6 rows change.
- **The midpoint issue:** 5.5 assumes a 1–10 scale, but the data are 0–10 and rest at about 4.9. H6
  stays as written.
  - **Secondary, labelled as added after seeing the scale:** the same test with intensity measured
    from the person's own resting position, |a − a_rest|, where a_rest is the median slider value
    over their baseline.

### B. The paper's evaluation, adapted (exploratory: it judges the state against self-report and cannot change a verdict)

- **Participants:** the 41 whose slider moves at least 0.5 across the films. The 9 above are left
  out; that was decided from the slider alone.
- **Slider intensity:** I_k = |a_k − a_rest|, scaled per person to fill 0–100: I′ = 100·I / (the
  person's maximum I over film bins). This follows the paper's per-person scaling.
- **HAI:** HAI_k = Φ((x̂_k − m) / √v_k), where m is the median of x̂ over the whole session.
- **Windows:** non-overlapping 20 s windows inside each film clip, 6 per clip and 36 per person.
  Each window gets the mean state, the mean HAI and the mean I′.
  - **Primary: no lag,** as in the paper.
  - **Secondary: the slider is read 5 s later** (bin k + 5, within the same clip), because a rating
    trails the feeling.

**Analysis 1, high/low agreement.** HAI-high = the person's 6 windows with the highest mean HAI (the
top 15%, rounded up). Slider-high is defined three ways:
- **(i) Primary, rank-matched:** the person's 6 windows with the highest I′. Nothing is tuned.
  Chance overlap is 1 window in 6.
- **(ii) Secondary, one cutoff on I′ chosen by leave-one-participant-out:** for each person, the
  cutoff c ∈ {0, 1, …, 100} with the best median **balanced** accuracy ((sensitivity +
  specificity)/2) over the other 40, applied to that person.
  - *Amended before any run:* plain accuracy would pick "everything low", which scores 30/36 =
    0.83 for nothing. That 0.83 is reported as the reference for every accuracy.
- **(iii) The paper's in-sample per-person tuning:** c ∈ {50, …, 100}, best accuracy with at least
  5 slider-high windows. It is reported only to compare with the paper's 69%, and it is optimistic
  by construction.

Reported for each: summed TP, TN, FP and FN, and the median per-person accuracy, sensitivity and
specificity.

**Analysis 2, direction of change.** Consecutive film windows in viewing order.
- Each window gets a level from I′:
  - **two levels:** split at the person's median;
  - **three levels:** split at the 20th and 80th percentiles (the paper's example).
- A transition is a pair of consecutive windows at different levels. It is a success if the mean
  state moves the same way as I′.
- Reported: accuracy pooled over all transitions, and for three levels split into low↔high and
  transitions involving mid.
- **Clip-level secondary:** the 5 consecutive clip-to-clip changes per person, all counted.

**Chance.** Windows next to each other are not independent, so the null distribution comes from
2,000 circular shifts of each person's slider-window sequence against their state windows (seed
19). The cutoff in (ii) is kept at its real value. "Agreement above chance" means p < 0.05 for:
- analysis 1 (i): the median accuracy;
- analysis 2: the pooled accuracy, two levels and three levels.

**States:**
- **Primary:** the main analysis's chosen state (`data/stage2_states_005.csv`).
- **For comparison:** tonic14c (Part 18), run through the same steps. It changes no decision.

**Self-test, before the real run:**
- a slider built from the state itself must give accuracy far above chance;
- a slider built from an unrelated random walk must give about chance.

**Outputs:** `s2_slider_eval` → `data/stage2_slider_eval.csv`, `data/stage2_slider_windows.csv`
and `results/figures/stage2_slider_eval.png`.

**Self-test: pass** (300 shifts each).

| Self-test | Analysis 1 (i) | Analysis 1 (ii), balanced | Analysis 2, direction (all versions) |
|---|---|---|---|
| Slider = the state | 1.00 (p 0.003) | 0.92 (p 0.003) | 0.96–1.00 (p 0.003) |
| Slider = a random walk | 0.71, the null mean (p 0.76) | 0.50 (p 1) | 0.45–0.50 (p 0.52–0.96) |

- **Three-level low↔high transitions are rare:** 2–4 in the whole sample. Consecutive 20 s windows
  seldom jump from the bottom 20% to the top 20%. The three-level result therefore rests almost
  entirely on transitions involving mid.
- **A leak, recorded:** the self-test run also printed the "H6 as Part 14 on these 41" row computed
  from the real slider (r 0.12, p 0.11). That row did not switch to the test slider. It was seen
  before the planned run. The code now skips that row in a self-test. Every analysis and threshold
  above was already fixed in writing, so nothing was changed because of it.

### Results (2026-11-23, run once)

**A. H6 as set in Part 14** (`s2_tests(pwd)` → `data/stage2_tests.csv`):
- **Part 14's other results are untouched:** all 168 non-H6 rows and the block summary are
  identical to the backup.

| | Participants | Mean Spearman r [95% CI] | p (rank p) | Outcome |
|---|---|---|---|---|
| **H6:** state vs \|a − 5.5\| across the 6 clips | 42 of 50 (slider varies) | +0.14 [−0.01, +0.28] | 0.074 (0.067) | **not supported** |
| State vs raw valence (secondary) | 45 | −0.04 [−0.21, +0.12] | 0.60 | reported |
| H6 secondary, \|a − a_rest\| (`s2_slider_eval`) | 41 | +0.05 [−0.13, +0.23] | 0.56 | reported |

- **H6 is not supported.** The direction is positive, with a 90% interval of [+0.01, +0.26], but
  p is 0.074.
- **Measuring intensity from each person's resting position** (the scale fix) weakens it rather
  than strengthening it.

**B. The paper's evaluation, main analysis's state** (`data/stage2_slider_eval.csv`,
`results/figures/stage2_slider_eval.png`; 41 participants, 2,000 shifts):

| | No lag | Slider read 5 s later |
|---|---|---|
| 1 (i) rank-matched, median accuracy (null mean) | 0.714 (0.711), p 0.90 | 0.714 (0.710), p 0.89 |
| 1 (i) sensitivity / specificity (median) | 0.17 / 0.83 | 0.17 / 0.83 |
| 1 (ii) LOPO cutoff, median balanced accuracy | 0.517, p 0.22 (cutoff median 14) | 0.533, p 0.055 |
| 1 (iii) in-sample cutoff (the paper's way), median accuracy | 0.714 (cutoff median 54) | 0.722 |
| **2 direction, two levels** | **0.555 of 236, p 0.025** | 0.534 of 249, p 0.097 |
| **2 direction, three levels** | **0.543 of 289, p 0.048** | 0.505 of 287, p 0.38 |
| 3 levels: low↔high / involving mid | 0.67 of 9 / 0.54 of 280 | 0.50 of 20 / 0.51 of 267 |
| 2 direction, clip to clip | 0.547 of 181, p 0.019 | |

**Comparison state, tonic14c** (`data/stage2_slider_eval_tonic14c.csv`):
- High/low agreement: the same as the main state (no agreement).
- Direction, two levels: **0.589, p 0.002**.
- Direction, three levels: 0.547, p 0.043.
- Clip to clip: 0.514, p 0.16.

**What it shows:**
- **The state does not pick out a person's highest-slider periods.** The person's 6 top-HAI windows
  overlap their 6 top-slider windows by 1, which is chance.
  - The paper's in-sample method gives 0.71 here, close to the paper's 69%. That number is also
    what chance gives (rank-matched null 0.71), and below "everything low" (0.83). The paper's
    accuracy should not be read as agreement on its own.
- **The state tends to move in the same direction as the slider, slightly but above chance:**
  - 55% of transitions with two levels (p 0.025), 54% with three levels (p 0.048), and 55% from
    clip to clip (p 0.019);
  - this is within the paper's 54.5–63.6% range;
  - it is weaker when the slider is read 5 s later. The secondary lag was a guess and was not
    tuned.
- **The tonic14c state does as well or slightly better on direction** (59% with two levels). It
  is the comparison state only and changes no decision.
- **How strong this is:** four of the six primary-ish tests pass, at p between 0.002 and 0.05.
  No correction across them was set in advance. Together with H6 (r 0.14) and the paper's own
  R² of 0.03, the agreement between self-report and the pulse-based state is real but small.
- **Limits:**
  - the slider is valence; intensity is a stand-in for arousal, and sadness and tenderness are
    clearly valenced but low in arousal;
  - 9 of 50 hardly use the slider;
  - the windows within a person are dependent, which is why the null is a circular shift.

**Conclusion for the report:**
- **H6 is not supported** (r 0.14, p 0.074).
- **The paper's evaluation, adapted:** the decoded state follows the *direction* of changes in
  self-reported intensity slightly above chance (54–59%), at the same level the paper reports, but
  it does not identify the *moments* of highest self-reported intensity.
- **The slider is used only to evaluate the state, as in the paper.** Using it as a model input
  would make this evaluation circular (Part 19, B, and the discussion before it).

## Current status

**Frozen setting: `own4_p1000_k14_f3_s8`** (Part 9). 4 Hz input with a 1 Hz low-pass; prior weight
2·10⁻⁴ (the paper's 0.1 weakened 1000×, doubled at 4 Hz); knots every 14 s; phase-1 λ limit 2·10⁻³;
GCV limits 2·10⁻⁴ (λ2) and 2·10⁻⁴ (λ1); one impulse per signal sample; 8 random starts; in µS.
Everything else as in `own_config.m`. On the 50 unseen test sessions: F1 0.82, recall 0.85,
precision 0.85 for every pulse ≥ 0.02 µS; F1 0.96, recall 0.98, precision 0.96 for pulses ≥ 0.05 µS.

Limitations to report:

1. Slow responses (half-recovery > 7 s): decay 37% short and precision 0.74, from false pulses
   along their tails (Part 9). During dense slow responding the true tonic is never visible, so
   the tonic is placed too high and the decays too short; neither the prior (round 7) nor the
   knot spacing (round 8) changes it.
2. On fast and typical shapes the decays run long instead (test +12% on fast, re-synthesis +26%,
   waveforms ×1.19), so the real decays (median 3.4 s) are likely overestimated (Parts 10, 11).
3. False pulses where the tonic rises faster than its knots allow (a stiffer tonic makes more of
   them, round 8).
4. True rises under about 0.45 s overestimated, near what a 4 Hz signal resolves (round 3).
5. The generator's limits (Part 8), and the real signals' fast symmetric transients it lacks
   (Part 11).
6. Generator and solver share the same response model, which favours model-based methods in any
   comparison on synthetic data.
7. The first round-6 run could not be scored (its truth was overwritten).
8. On real data about 20% more pulses than the synthetic tests predict, as small doublets on
   response rises (Part 12); about half are below 0.05 µS.

## Next steps

1. **Finish checking that the real pulses make sense before stage 2**, blind to the hypotheses.
   Done: the count (Part 12). Left: none in flat stretches, sizes against the counted SCRs,
   plausible shapes per participant. The settings are frozen, so this cannot feed back into
   tuning, and H1–H6 are not looked at.
2. **Compare with other methods** under the same conditions (the V test sessions and the real 50,
   same input, same scoring): cvxEDA and the authors' published solver (branch `black-box`), each
   at its published defaults or tuned the same way, stated. On real data only fit, sparsity and
   plausibility can be compared, since there is no truth.
3. Stage 2 (impulses → arousal state) and hypotheses H1–H6. **Threshold, fixed 2026-11-14 before
   H1–H6 were looked at, on accuracy alone:** the main analysis counts pulses ≥ 0.05 µS (test F1
   0.96, recall 0.98, precision 0.96); all pulses ≥ 0.02 µS (F1 0.82) are the secondary check. Input:
   `data/real50_pulses.csv`. The first stage-2 code is on `black-box` (`state.py`) and comes over
   then.
   **First descriptive look (2026-11-14):** pulse rates by block and film group (`block-rates`,
   `data/real50_block_rates.csv`), taken after the solver, the selection and the threshold were
   frozen, with no tests. Nothing in stage 1 changes after it; the stage-2 tests are set before
   they are run.
   What it showed (pulses ≥ 0.05 µS per minute, medians; baseline → own films; within-person
   films / baseline): neutral group 10.8 → 9.2 (0.77), negative 10.4 → 10.1 (0.93), positive 12.3 →
   12.3 (1.01), mixed 7.4 → negative 9.8 and positive 7.2 (1.21). Every block has pulses at
   plausible rates. Groups already differ at baseline and the baseline comes first (responding
   fades over a session), so stage 2 compares films with each person's own baseline.
   Pulse size by block (median height, and summed heights per minute) was added to `block-rates`
   after these rates had been seen, at the user's request: recorded here so that any use of size in
   stage 2 is known to have been chosen after looking.
   What it showed (pulses ≥ 0.05 µS, within-person films / baseline, medians; rate, size, rate ×
   size): neutral 0.77, 1.01, 0.81; negative 0.93, 1.16, 1.14; positive 1.01, 0.99, 1.01; mixed
   1.21, 1.09, 1.34. Size rises only where negative films are shown; films are above baseline in
   rate × size only in the groups with negative films. Groups of 10, no tests.
   The timeline (POPANE, Behnke et al. 2022): a 5-minute baseline, an unlabelled transition (median
   150 s, 96–338 s; its content is not described in the paper, probably on-screen instructions),
   then 12 minutes of film clips back to back (six clips; mixed group: 3 negative and 3 positive,
   corrected 2026-11-16 from "6 negative, 6 positive", Part 14), then an
   ultimatum game, which our segment excludes. The transition is left out of the block rates, and
   it adds to the time between baseline and films.
   **Stage 2 state estimation done (Part 13, 2026-11-16):** the Data Challenge 2 filters, validated
   on 400 synthetic sessions (C3 failed as first set; C3′, rescored after the result, passed), run on
   the 50. **H1–H6 tests set in writing, then run once (Part 14, 2026-11-16):** H2 supported; H1,
   H3 and H4 not supported; within the mixed group negative and positive are equivalent; H5 no
   evidence of separation; H6 not supported (run 2026-11-23 once the raw files were in, r +0.14,
   p 0.074; Part 19, with the slider evaluation of Oliver et al. 2025). Exploratory tonic variants
   in Parts 15–18. Next: the report and slides.
4. Report and slides.

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
4. Judge the round session by session, whatever the verdict: set its control and variants in
   `src/eda/tuning.py` (`COMPARE_*`), then `PYTHONPATH=src python -m eda.run round-compare`: pulses
   found, missed and false by size and closeness, the tonic's share of the responses, the rule set
   before the run, and one figure page per session with every setting's tonic and pulses
   (`data/round<k>_compare.csv`, `results/figures/round<k>_tonic.pdf`).

Before any code change: `own_test` in MATLAB (`addpath src/own; own_test`).

## Where everything is

| What | Where |
|---|---|
| **Stage 2's input:** every real pulse and window fit | `data/real50_pulses.csv`, `data/real50_windows.csv` (`real-pulses`) |
| Plausibility of the real pulses (Part 12) | `data/pulse_check_isolated.csv` (`pulse-check`) |
| First descriptive look at blocks and film groups | `data/real50_block_rates.csv` (`block-rates`) |
| Per-session comparison of a round's settings | `data/round<k>_compare.csv`, `results/figures/round<k>_tonic.pdf` (`round-compare`) |
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

## Sources checked (2026-11-14)

- Amin & Faghih 2021, IEEE TBME 68(5):1726–1736, accepted manuscript: https://par.nsf.gov/servlets/purl/10220906
- Alam, Amin & Faghih 2023, IEEE OJEMB: https://pmc.ncbi.nlm.nih.gov/articles/PMC10776104/
- Amin & Faghih 2022, PLOS Comput Biol: https://pmc.ncbi.nlm.nih.gov/articles/PMC9333288/
- Greco et al. 2016, cvxEDA, IEEE TBME 63(4):797–804; code: https://github.com/lciti/cvxEDA
- Benedek & Kaernbach 2010, J Neurosci Methods 190:80–91: https://pmc.ncbi.nlm.nih.gov/articles/PMC2892750/
- Benedek & Kaernbach 2010, Psychophysiology 47:647–658: https://pmc.ncbi.nlm.nih.gov/articles/PMC2904901/
- NeuroKit2 `eda_phasic`: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/eda/eda_phasic.py
- Kurinec et al. 2022, Front Behav Neurosci (resting NS-SCR rate 1–3 per minute, citing Dawson et al. 2017): https://www.frontiersin.org/articles/10.3389/fnbeh.2022.885302/full
- Gertler et al. 2020, NeuroImage (about 5, up to 10 per minute, citing Zimmer 2000; 6.7 per minute above 0.1 µS at rest): https://pmc.ncbi.nlm.nih.gov/articles/PMC7263940/
- Behnke et al. 2022, Scientific Data 9:10 (POPANE, Study 6 procedure; affect slider is valence, 1 extremely negative to 10 extremely positive, read 2026-11-16): https://pmc.ncbi.nlm.nih.gov/articles/PMC8776805/
