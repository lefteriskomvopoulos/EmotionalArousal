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
#   own4_p100_k14_f3: ,${base},'prior_lambda',[0.002;0.002]
# Round 4, suffix=own4: the prior 10x weaker (p1000, 1e-4 doubled) and off (p0).  p1000 won.
suffix=own4
base="'u_per_y',1,'knot_s',14,'lambda2_max_focuss',2e-3,'lambda2_max_gcv',2e-4,'lambda1_max',2e-4"
settings=(
  "own4_p1000_k14_f3:,${base},'prior_lambda',[0.0002;0.0002]"
  "own4_p0_k14_f3:,${base},'use_prior',false"
)

: > "$b/jobs.txt"
for s in $(awk -F, '$3 == "tune" {print $1}' "$root/data/own_split.csv"); do
  cp "$root/data/tbme_input/${s}_${suffix}.mat" "$b/input/"
  for e in "${settings[@]}"; do
    echo "'${s}_${suffix}','amin2021','tag','${e%%:*}'${e#*:}" >> "$b/jobs.txt"
  done
done
n=$(wc -l < "$b/jobs.txt" | tr -d ' ')
sed -i '' "s/^#SBATCH --array=.*/#SBATCH --array=1-$n/" "$b/run_own.sbatch"
(cd "$root/hpc" && zip -qr own_bundle.zip own_bundle)
echo "bundle: $(ls "$b/input" | wc -l | tr -d ' ') sessions, $n jobs -> $root/hpc/own_bundle.zip"
