# Visual inspection: 18 preview tuning sessions against their real participants

Generator: synth-calibrate (rate x1.45, size x0.70, spread x1.06 of ln(p90/median)/z90, noise x0.81),
tune50-preview, 2026-11-11.  Pages: tune50_preview_vs_real.pdf.  Real / synthetic counted numbers in
each page title.

| session | tau group | what differs from the real participant |
|---|---|---|
| U127 | far | film responses larger than baseline in real, not in synthetic; far decays broader (design); rate 0.74 |
| U173 | far | sizes about 2x (p90 1.58 vs 0.70): wide spread (1.52) plus 5 s decays merging |
| U53 | far | film-onset burst (film 105) missing; responses broader (design); size 1.3x |
| U135 | far | good; onsets abrupt |
| U12 | far | good; tonic plateau matched |
| U91 | far | real film responses 5x baseline (median 0.26 vs 0.05 uS); synthetic the same everywhere |
| U67 | moderate | baseline spikes too large; dense film activity (12-15 min) became a smooth floor hump; rate 0.63 |
| U118 | moderate | film-onset response (305) missing, replaced by a smooth floor hump |
| U48 | moderate | 0.38 / 2.08 s: needle-like spikes, narrower than any real response |
| U68 | moderate | bimodal sizes (median 0.03, p90 0.55): synthetic median 3.4x, p90 2.7x |
| U89 | moderate | 0.37 / 2.07 s: needles against broad real responses |
| U78 | moderate | the 2.1 uS film-onset response missing; small scattered ones instead |
| U165 | prior | good |
| U61 | prior | spread 1.73: baseline full of 2-3 uS spikes; p90 1.66x |
| U138 | prior | real responses broader with long recovery |
| U178 | prior | film-onset responses missing; floor ramps |
| U137 | prior | good |
| U161 | prior | baseline responses too large (real 0.06 vs film 0.11 uS); film-onset jump became a 30 s ramp |

## Causes

1. Size does not depend on the block.  Real film responses are larger than baseline ones (median
   film / baseline: P91 0.26 / 0.05, P161 0.11 / 0.06, P67 0.06 / 0.04 uS); the generator varies
   only the rate by block.
2. The counting limit is corrected for the cohort, not per participant.  Where the counted median
   sits near 0.02 uS (P68 0.03), most of the true distribution is hidden below it, and the cohort
   size factor oversizes the session (U68, U61, U173).
3. Film-onset responses.  The largest real responses often come at a block start (orienting
   responses); synthetic pulses fall at random times, and the floor (5th percentile over 30 s,
   smoothed over 30 s) turns the real onset jump into a smooth ramp or hump.
4. Shape.  The moderate group's fast variant (0.4 / 2.0 s, peak after 0.8 s) is faster than real
   SCR rise times (1-3 s); with the Bateman onset at full slope it gives needles.
5. Rate outside baseline and films (the gap between them) is set to the overall rate, not the
   participant's.

# Second inspection: the frozen candidate (2026-11-11)

Generator: each participant's counted responses resampled per block at random times, real-data
shapes from another participant, 60 s tonic, slow fluctuations matched to the small-SCR count,
noise matched to roughness (own_preprocess.flat_session; synth-check, tune50-preview).

| weakness | where | why |
|---|---|---|
| slow tonic ramps (about 1 min) with no response at their start | U118 10.3 min, U12 13.8 / 17.5, U173 11.5, U161 10.8 | a sharp real rise followed by a plateau of minutes; any windowed tonic turns it into a ramp |
| shapes unlike the participant's | U61 broad (rise 2.0 s, at the solver's bound) against sharp real spikes; U48 symmetric 1.5 / 1.6 s against slower real recoveries | each session draws a shape from another participant; the pool holds odd fits: decay < 1.5x rise in 87 of 417 windows, rise at the 2.0 s bound in 7 |
| activity flat across films | U48 (very active first films, quiet after 17 min), U118, U161 | one rate per block kind; real responding varies film to film and fades over the session |
| large responses larger and fewer | U48 baseline median 0.48 vs 0.25 uS, U138 p90 0.81 vs 0.55 | counted responses are already merged; resampled and overlapped again, they merge twice |
| long merged humps in real baselines missing | U138 baseline (6.5-8 min) | dense real bursts at specific moments; random timing spreads them |

# Third inspection: frozen (2026-11-11)

Shape pool filtered (decay >= 2x rise, off the solver's bounds: 270 windows of 49 participants,
rise median 0.94 s, decay 3.96 s) and a rate per run (each film its own): U48 now active in the
first films and quiet after 17 min like P48, U61 sharp like P61.  synth-check, generated / real:
roughness 1.03, small SCRs 0.98, median size 1.07, p90 1.10, block sizes 1.12 / 1.13, clustering
0.91, all counted 0.84, large (>= 0.05 uS) 0.78.  Left as limitations: tonic ramps at large onset
events, large responses fewer and larger (counted responses merge twice), dense real bursts at
specific moments (random timing).

Final change: each run reuses its own counted responses, each once in shuffled order (was: drawn
with replacement from the whole block), so a session holds its participant's large responses in
the film where they occurred (U118: the 1.98 uS response in film 305).  synth-check: roughness
1.03, small 0.96, median size 1.07, p90 1.14, block sizes 1.16 / 1.14, clustering 0.92, all
counted 0.85, large 0.82; slow power missing from flat sessions 1.3e-07 uS^2 (slow-noise
versions dropped from round 6).

Shapes, final: one per session, drawn at random within physiological limits (peak 1-3 s after
onset, half-recovery 2-10 s, decay >= 2x rise, solver bounds), not from the participant or the
solver; tuning set stratified 6 fast (< 4 s) / 6 typical / 6 slow (> 7 s).  synth-check: roughness
1.02, small 1.00, median size 1.05, p90 1.11, block sizes 1.14 / 1.14, clustering 0.95, all
counted 0.87, large 0.81; slow power missing 0.  Generator frozen.

Fluctuations matched on the solver input (was: plain 2 Hz, where real small counts include fast
ripples the 1 Hz filter removes; tuning sessions then had typical size 0.067 uS on the solver
input against the real 0.094).  synth-check on the solver input: small SCRs 1.01, typical size
0.90, roughness 1.04; own-scale: tuning sessions 0.099 uS against real 0.094.  On the plain 2 Hz
signal the small counts are now 0.71 of real (the fast ripples, which the solver never sees).
Bundle: 18 sessions, 90 jobs.  Frozen.
