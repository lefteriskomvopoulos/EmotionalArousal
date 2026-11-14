#!/bin/bash
# Build hpc/own_bundle(.zip): the 18 tuning sessions under every setting below.
# Test sessions are left out on purpose.  Needs: python -m eda.run own-export (or own4-export)
# Each entry is  tag:settings  where settings are extra arguments of own_run.m.
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
b="$root/hpc/own_bundle"
rm -rf "$b" "$root/hpc/own_bundle.zip"
mkdir -p "$b/input"
cp "$root"/src/own/*.m "$root/src/own/run_own.sbatch" "$b/"

# Every round so far, on the 18 tuning sessions (results in hpc/own/tune/output/<tag>):
#
# Round 1, suffix=own (0.5 Hz, 2 Hz).  Knot 14 won.
#   p1_k6:      (own_config defaults)
#   p10_k6:     ,'prior_lambda',[0.01;0.01]
#   p100_k6:    ,'prior_lambda',[0.001;0.001]
#   p1_k14:     ,'knot_s',14
#   p10_k14:    ,'prior_lambda',[0.01;0.01],'knot_s',14
#   p100_k14:   ,'prior_lambda',[0.001;0.001],'knot_s',14
# Round 2, suffix=own: phase-1 lambda limit.  p100_k14_f3 won.
#   p10_k14_f3:  ,'prior_lambda',[0.01;0.01],'knot_s',14,'lambda2_max_focuss',1e-3
#   p10_k14_f4:  ,'prior_lambda',[0.01;0.01],'knot_s',14,'lambda2_max_focuss',1e-4
#   p100_k14_f3: ,'prior_lambda',[0.001;0.001],'knot_s',14,'lambda2_max_focuss',1e-3
#   p100_k14_f4: ,'prior_lambda',[0.001;0.001],'knot_s',14,'lambda2_max_focuss',1e-4
# Round 3, suffix=own4 (1 Hz, 4 Hz): p100_k14_f3 on the new input.  u stays at 4 Hz
# (u_per_y 1), so u means the same as before.  Twice the samples doubles the data term, so every
# penalty weight is doubled to keep its balance with the fit: prior, both sparsity limits, tonic
# limit.  own4 won (F1 .85 vs .81, rise floor down), decay still pulled toward the prior's 2.77 s.
#   own4_p100_k14_f3: ,${base},'knot_s',14,'prior_lambda',[0.002;0.002]
# Round 4, suffix=own4: the prior 10x weaker (p1000, 1e-4 doubled) and off (p0).  p1000 won.
#   own4_p1000_k14_f3: ,${base},'knot_s',14,'prior_lambda',[0.0002;0.0002]
#   own4_p0_k14_f3:    ,${base},'knot_s',14,'use_prior',false
# Round 5, suffix=own4: tonic knot spacing 10 and 20 s at the best setting (14 s is round 4's).
# Decays of 2-4 s came out 10-35% too long: a too-stiff tonic leaves slow drift for a longer decay.
# 14 s stayed (own4_p1000_k14_f3, frozen for the final run).
#   own4_p1000_k10_f3: ,${base},'knot_s',10,'prior_lambda',[0.0002;0.0002]
#   own4_p1000_k20_f3: ,${base},'knot_s',20,'prior_lambda',[0.0002;0.0002]
#
# Round 6, the scale round.  The solver works in uS, and the penalties do not grow with the signal
# as the fit does (sparsity ~ size^0.5, prior ~ 1, fit ~ size^2), while GCV's lambda2 stopped at
# its limit in 91% of real windows against 38% in tuning.  New tuning sessions from here on:
# 18 built from the 50 selected participants (U<id>, tune50-export, data/tune50_cohort.csv): each
# reuses its participant's counted responses per run, with one random physiological shape, slow
# fluctuations matched to its small-SCR count and noise matched to its roughness (synth-check).
# Only own4: the frozen generator's flat sessions already hold the slow power of real quiet
# stretches (synth-check: 1.3e-07 uS^2 missing, against 5.1e-05 first), so own4s would repeat own4.
#   A  <suffix>_p1000_k14_f3_s8       the frozen setting with 8 starts (the paper's number)
#   B  <suffix>_p1000_k14_f3_s8_g10   A with the GCV sparsity limit 10x higher, so GCV can choose
#   C  <suffix>_p1000_k14_f3_s8_norm  A with each session scaled to the tuning sessions' typical
#                                     response size (y_gain from data/own_scale.csv, own-scale)
#   scale check, own4 only: A with y x 0.5 and x 2 (_s8_x05, _s8_x2).  Noise scales with the
#   pulses, so any change in the scores is the penalties' dependence on scale alone.
# Result (2026-11-12, median of the 18, f1_visible / recall / precision): A .835 / .850 / .816,
# B .827 / .850 / .819 (lambda2 at its limit 0% against 45%: the limit does not cost accuracy),
# C .792 / .836 / .806 (worse: noise is about constant in uS, so scaling small responders up
# amplifies it), x0.5 .815 / .776 / .883, x2 .784 / .900 / .729 (scale shifts recall against
# precision; x1 is best).  Slow shapes (half-recovery > 7 s): decay about 40% short.
# Frozen: A, own4_p1000_k14_f3_s8.
#   (round 6 jobs per session: ${base}; ${base},'lambda2_max_gcv',2e-3; ${base},'y_gain',<own-scale>;
#    and on own4 ${base},'y_gain',0.5 / 2)
#
# Round 7, the prior on rise and decay.  On the final run's test sessions, slow shapes lost
# precision to false pulses along long recoveries fitted too short (decay -37%).  The paper's prior
# (0.65 +- 0.21 s rise, 2.77 +- 0.52 s decay) is a normal fit to its previous study's estimates
# (Amin & Faghih 2019: auditory stimulation, 26 participants); on our real 50, 47% of the frozen
# run's windows fall outside its +-2 SD (data/decay_vs_prior.csv).  One variant of A, same 18:
#   A_free   no prior on rise or decay, only the bounds        (_free)
# Control: A itself (round 6, own4_p1000_k14_f3_s8).  A prior refitted to our own estimates on the
# real 50 (data/prior_from_real50.csv) was considered and dropped: it would feed the solver's own,
# biased outputs back into it (the paper's prior comes from a different dataset).  No prior matches
# the generator, which draws shapes uniformly within physiological limits.
# A model-free prior measured on the waveforms (scr-shapes, 2026-11-12) was dropped too: on the
# synthetic test sessions it failed the criterion set before the run for recovery (+12.9%) and
# decay (+13.9%; rise passed, -0.1%, Spearman .97), and 36% of real isolated SCRs recover as fast
# as they rise, which no kernel shape fits.  Refining it until it passes would be tuning the
# measurement to the check.  Kept as a description only (data/prior_measured50.csv).
# Result (2026-11-13, median of the 18, against A): F1 visible .825 vs .835, precision .812 vs .816,
# amplitude rank .947 vs .971, windows meeting the authors' rule .83 vs 1.00 (slow shapes .36 vs
# .83); slow decay -28% vs -41% but slow precision .736 vs .767.  No better and less stable, as in
# round 4: A stays, nothing re-run.  The prior is not what makes the false pulses on slow tails.
#   (round 7 job per session: ${base},'use_prior',false, tag own4_p1000_k14_f3_s8_free)
#
# Round 8, the tonic's flexibility.  In A, GCV's tonic penalty lambda1 is at its limit in only
# 7-12% of the U windows and is chosen near zero (1e-8 to 1e-6 on P53), so the tonic's shape is set
# by the knot spacing alone.  On slow shapes it takes 18% of each response (phasic_taken .18,
# about 0 on fast and typical): a spline with knots every 14 s bends into a 10 s recovery.  Round 5
# tested spacing on the first 18 (S) only.  Two variants of A, same 18 U:
#   A_k20   knots every 20 s                                   (own4_p1000_k20_f3_s8)
#   A_k28   knots every 28 s, twice A's                        (own4_p1000_k28_f3_s8)
# Control: A (round 6).  Rule, set before the run: a variant replaces A only if slow-shape precision
# and phasic_taken both improve, and F1 visible on fast and typical shapes falls by no more than
# .01 each.  Otherwise A stays.  A floor on lambda1 was considered and dropped: the penalty is a
# ridge on the spline coefficients (the tonic's level, in uS), so a floor pulls the tonic down
# under the constraint C q <= y and hands more of the signal to pulses; it does not stiffen the
# shape.  If neither spacing passes, the next candidate is a penalty on the tonic's curvature
# (P-spline, Eilers & Marx 1996), a change to the paper's method.
# Result (2026-11-14, median of the 18; round-compare, data/round8_compare.csv): F1 visible .835 /
# .803 / .794 (14 / 20 / 28 s), precision .816 / .786 / .760; slow precision .767 / .722 / .631,
# slow phasic_taken .184 / .198 / .185.  Both fail: wider knots miss 59-68 fewer true pulses but
# return 115-287 more false ones, nearly all standing alone.  On slow sessions the estimated tonic
# sits 0.2-0.4 uS above the true one at every spacing (an offset, not a bend: overlapping long
# recoveries hide the floor), so the curvature penalty is not run either.  A stays, nothing re-run.
#   (round 8 jobs per session: ${base},'knot_s',20 and ${base},'knot_s',28)
base="'u_per_y',1,'lambda2_max_focuss',2e-3,'lambda2_max_gcv',2e-4,'lambda1_max',2e-4"
base="${base},'knot_s',14,'prior_lambda',[0.0002;0.0002],'n_starts',8"

: > "$b/jobs.txt"
sessions=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next} {print $h["session"]}' \
           "$root/data/tune50_cohort.csv")
for s in $sessions; do
  n="${s}_own4"
  cp "$root/data/tbme_input/${n}.mat" "$b/input/"
  for k in 20 28; do
    echo "'${n}','amin2021','tag','own4_p1000_k${k}_f3_s8',${base},'knot_s',${k}" >> "$b/jobs.txt"
  done
done
n=$(wc -l < "$b/jobs.txt" | tr -d ' ')
sed -i '' "s/^#SBATCH --array=.*/#SBATCH --array=1-$n/" "$b/run_own.sbatch"
(cd "$root/hpc" && zip -qr own_bundle.zip own_bundle)
echo "bundle: $(ls "$b/input" | wc -l | tr -d ' ') sessions, $n jobs -> $root/hpc/own_bundle.zip"
