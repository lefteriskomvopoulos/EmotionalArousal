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

# Round 1 (done, in hpc/own/tune/output): p1/p10/p100 x knot 6/14.  Knot 14 won.
# Round 2 (done): phase-1 lambda limit (lambda2_max_focuss) 1e-3 and 1e-4.  p100_k14_f3 won.
# Round 3 (done): p100_k14_f3 on the own4 input (1 Hz low-pass, 4 Hz).  u stays at 4 Hz
# (u_per_y 1), so u means the same as before.  Twice the samples doubles the data term, so every
# penalty weight is doubled to keep its balance with the fit: prior, both sparsity limits, tonic
# limit.  own4 won (F1 .85 vs .81, rise floor down), decay still pulled toward the prior's 2.77 s.
# Round 4: the prior 10x weaker (p1000, 1e-4 doubled) and off (p0), everything else as round 3.
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
