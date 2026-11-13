#!/bin/bash
# Build hpc/own_final_bundle(.zip): the final run, the frozen setting once on the 50 selected
# participants (data/selection.csv): their real inputs P<id>_own4 and their test sessions
# V<id>_own4 (the round-6 generator, own seed, never used in tuning).  100 jobs.
# Needs: python -m eda.run final-export (the real inputs) and test50-export (the test sessions).
# Results go to output/own4_p1000_k14_f3_s8 on Torch; locally into hpc/own/final/output.
#
# The first final run (frozen after round 5, 150 jobs: P, F, T) used tag own4_p1000_k14_f3 with
# the args below minus 'n_starts',8, and kinds "P F T"; its results stay in
# hpc/own/final/output/own4_p1000_k14_f3 (final-score-first).
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
b="$root/hpc/own_final_bundle"
rm -rf "$b" "$root/hpc/own_final_bundle.zip"
mkdir -p "$b/input"
cp "$root"/src/own/*.m "$root/src/own/run_own.sbatch" "$b/"

# Frozen after tuning round 6 (make_own_bundle.sh): own4 input, prior 1e-4 (doubled at 4 Hz),
# knots every 14 s, phase-1 sparsity limit 1e-3 (doubled), every penalty doubled at 4 Hz, 8 starts,
# in uS (no normalization).
suffix=own4
tag=own4_p1000_k14_f3_s8
args="'u_per_y',1,'lambda2_max_focuss',2e-3,'lambda2_max_gcv',2e-4,'lambda1_max',2e-4,'knot_s',14,'prior_lambda',[0.0002;0.0002],'n_starts',8"

ids=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next} $h["selected"]=="True"{print $h["id"]}' \
      "$root/data/selection.csv")
: > "$b/jobs.txt"
for kind in P V; do
  for id in $ids; do
    cp "$root/data/tbme_input/${kind}${id}_${suffix}.mat" "$b/input/"
    echo "'${kind}${id}_${suffix}','amin2021','tag','${tag}',${args}" >> "$b/jobs.txt"
  done
done
n=$(wc -l < "$b/jobs.txt" | tr -d ' ')
sed -i '' "s/^#SBATCH --array=.*/#SBATCH --array=1-$n/" "$b/run_own.sbatch"
(cd "$root/hpc" && zip -qr own_final_bundle.zip own_final_bundle)
echo "bundle: $(ls "$b/input" | wc -l | tr -d ' ') inputs, $n jobs -> $root/hpc/own_final_bundle.zip"
