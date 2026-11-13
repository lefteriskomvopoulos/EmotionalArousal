#!/bin/bash
# Build hpc/own_resynth_bundle(.zip): the papers' re-synthesis test (Amin & Faghih 2019, 2021).
# The 50 real signals rebuilt from our final-run deconvolution plus white noise at each one's fit residual, R<id>_own4,
# deconvolved again with the frozen setting.  50 jobs.
# Needs: python -m eda.run resynth-export (after the final run's output is in hpc/own/final/output).
# Results go to output/own4_p1000_k14_f3_s8 on Torch; locally into hpc/own/resynth/output.
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
b="$root/hpc/own_resynth_bundle"
rm -rf "$b" "$root/hpc/own_resynth_bundle.zip"
mkdir -p "$b/input"
cp "$root"/src/own/*.m "$root/src/own/run_own.sbatch" "$b/"

# The frozen setting (make_own_final_bundle.sh).
suffix=own4
tag=own4_p1000_k14_f3_s8
args="'u_per_y',1,'lambda2_max_focuss',2e-3,'lambda2_max_gcv',2e-4,'lambda1_max',2e-4,'knot_s',14,'prior_lambda',[0.0002;0.0002],'n_starts',8"

ids=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next} $h["selected"]=="True"{print $h["id"]}' \
      "$root/data/selection.csv")
: > "$b/jobs.txt"
for id in $ids; do
  cp "$root/data/tbme_input/R${id}_${suffix}.mat" "$b/input/"
  echo "'R${id}_${suffix}','amin2021','tag','${tag}',${args}" >> "$b/jobs.txt"
done
n=$(wc -l < "$b/jobs.txt" | tr -d ' ')
sed -i '' "s/^#SBATCH --array=.*/#SBATCH --array=1-$n/" "$b/run_own.sbatch"
(cd "$root/hpc" && zip -qr own_resynth_bundle.zip own_resynth_bundle)
echo "bundle: $(ls "$b/input" | wc -l | tr -d ' ') inputs, $n jobs -> $root/hpc/own_resynth_bundle.zip"
