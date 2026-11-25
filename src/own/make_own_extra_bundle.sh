#!/bin/bash
# Build hpc/own_extra_bundle(.zip): the frozen setting, unchanged, once on the other 47 real
# participants, who passed both selection stages but were not drawn into the 50 (data/selection.csv:
# stage2_pass and not selected).  Exploratory, for stage 2; not part of the reported results or
# of H1-H6.  47 jobs, real inputs only (no test sessions).
# Needs: python -m eda.run extra-export (their inputs P<id>_own4).
# Results go to output/own4_p1000_k14_f3_s8 on Torch; locally into hpc/own/extra/output.
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
b="$root/hpc/own_extra_bundle"
rm -rf "$b" "$root/hpc/own_extra_bundle.zip"
mkdir -p "$b/input"
cp "$root"/src/own/*.m "$root/src/own/run_own.sbatch" "$b/"

# The frozen setting, read from the final run's script so that the two cannot differ.
final="$root/src/own/make_own_final_bundle.sh"
suffix=$(sed -n 's/^suffix=//p' "$final")
tag=$(sed -n 's/^tag=//p' "$final")
args=$(sed -n 's/^args="\(.*\)"$/\1/p' "$final")
[ -n "$suffix" ] && [ -n "$tag" ] && [ -n "$args" ] || { echo "could not read the setting from $final"; exit 1; }

ids=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next}
               $h["stage2_pass"]=="True" && $h["selected"]=="False"{print $h["id"]}' \
      "$root/data/selection.csv")
: > "$b/jobs.txt"
for id in $ids; do
  cp "$root/data/tbme_input/P${id}_${suffix}.mat" "$b/input/"
  echo "'P${id}_${suffix}','amin2021','tag','${tag}',${args}" >> "$b/jobs.txt"
done
n=$(wc -l < "$b/jobs.txt" | tr -d ' ')
sed -i '' "s/^#SBATCH --array=.*/#SBATCH --array=1-$n/" "$b/run_own.sbatch"
(cd "$root/hpc" && zip -qr own_extra_bundle.zip own_extra_bundle)
echo "bundle: $(ls "$b/input" | wc -l | tr -d ' ') inputs, $n jobs ($tag) -> $root/hpc/own_extra_bundle.zip"
