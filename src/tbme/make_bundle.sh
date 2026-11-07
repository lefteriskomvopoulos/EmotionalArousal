#!/bin/bash
# Build hpc/tbme_bundle(.zip): everything the cluster needs, nothing else.
# Usage: src/tbme/make_bundle.sh 99 5            real participants P99, P5
#        src/tbme/make_bundle.sh S99 S5          their synthetic sessions
#        src/tbme/make_bundle.sh all-synth       every synthetic session in data/tbme_input
#        src/tbme/make_bundle.sh all-shift       the level-shifted synthetic sessions (S<k>_m0/_m2)
#        src/tbme/make_bundle.sh S9:400 S9:800    a session with a non-default window length (s)
#        src/tbme/make_bundle.sh S9:-:1.27:5.14   tau fixed (+-5% bounds), default window
#        TIME=06:00:00 src/tbme/make_bundle.sh ...   a longer job time limit
#        src/tbme/make_bundle.sh all-f4          the synthetic sessions rebuilt at 4 Hz (S<k>_f4)
#        src/tbme/make_bundle.sh all-real        every real participant in data/tbme_input
# (inputs must be exported first: python -m eda.run profile / synth-tbme)
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
b="$root/hpc/tbme_bundle"
rm -rf "$b" "$root/hpc/tbme_bundle.zip"
mkdir -p "$b/input"
cp "$root/src/tbme/tbme_run.m" "$root/src/tbme/run_tbme.sbatch" "$b/"
cp -r "$root/external/Tonic-Phasic-Decomposition-TBME/Dependencies" "$b/"
: > "$b/pids.txt"
names=()
for p in "$@"; do
  case "$p" in
    all-synth) for f in "$root"/data/tbme_input/S*.mat; do n="$(basename "$f" .mat)"; case "$n" in *_*) ;; *) names+=("$n") ;; esac; done ;;
    all-shift) for f in "$root"/data/tbme_input/S*_m*.mat; do names+=("$(basename "$f" .mat)"); done ;;
    all-f4)    for f in "$root"/data/tbme_input/S*_f4.mat; do names+=("$(basename "$f" .mat)"); done ;;
    all-real)  for f in "$root"/data/tbme_input/P*.mat; do n="$(basename "$f" .mat)"; case "$n" in *_*) ;; *) names+=("$n") ;; esac; done ;;
    *:*)       names+=("${p//:/ }") ;;
    [0-9]*)    names+=("P$p") ;;
    *)         names+=("$p") ;;
  esac
done
for n in "${names[@]}"; do
  cp "$root/data/tbme_input/${n%% *}.mat" "$b/input/"
  echo "$n" >> "$b/pids.txt"
done
sed -i '' "s/^#SBATCH --array=.*/#SBATCH --array=1-${#names[@]}/" "$b/run_tbme.sbatch"
if [ -n "$TIME" ]; then sed -i '' "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" "$b/run_tbme.sbatch"; fi
(cd "$root/hpc" && zip -qr tbme_bundle.zip tbme_bundle)
echo "bundle: $b ($(ls "$b/input" | wc -l | tr -d ' ') sessions) -> $root/hpc/tbme_bundle.zip"
