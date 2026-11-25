#!/usr/bin/env bash
# Recompute the project's results and compare every file with the shipped one (REPRODUCE.md).
#
#   ./reproduce.sh                    everything that runs from the shipped files: no raw data, no cluster
#   ./reproduce.sh --fast             the same, without stage 2's synthetic validations (the slowest part)
#   ./reproduce.sh --rebuild          first rebuild every solver input and test truth from the raw POPANE
#                                     files (data/raw/Study6, or --raw DIR), then all of the above
#   ./reproduce.sh --solver V53_own4  also rerun the deconvolution on one session (P, V or R) and compare
#
#   --work DIR   the scratch copy the commands run in (default: $TMPDIR/physio_reproduce); the shipped
#                folder is never written to, except for reproduce_report.csv
#   --raw DIR    the POPANE Study 6 folder, if it is not at data/raw/Study6
#   PYTHON=..., MATLAB=...   the interpreters to use (default: .venv/bin/python, then python3; matlab)
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="${TMPDIR:-/tmp}"; WORK="${WORK%/}/physio_reproduce"
RAWDIR="$ROOT/data/raw/Study6"
FAST=0; REBUILD=0; SOLVER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --fast) FAST=1 ;;
        --rebuild) REBUILD=1 ;;
        --solver) SOLVER="${2:?--solver needs a session, for example V53_own4}"; shift ;;
        --work) WORK="${2:?--work needs a folder}"; shift ;;
        --raw) RAWDIR="${2:?--raw needs a folder}"; shift ;;
        -h|--help) sed -n 2,16p "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see ./reproduce.sh --help)"; exit 2 ;;
    esac
    shift
done

# ------------------------------------------------------------------------------------ tools
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY=python3; fi
if ! "$PY" -c "import numpy, scipy, pandas, matplotlib" 2>/dev/null; then
    echo "Python ($PY) lacks numpy, scipy, pandas or matplotlib. Set up as in REPRODUCE.md, Requirements."
    exit 2
fi
MATLAB="${MATLAB:-matlab}"
if command -v "$MATLAB" >/dev/null 2>&1; then HAVE_M=1; else HAVE_M=0; fi
command -v rsync >/dev/null 2>&1 || { echo "rsync is needed to make the scratch copy."; exit 2; }

HAVE_RAW=0
if ls "$RAWDIR"/S6_P*_All.csv >/dev/null 2>&1; then HAVE_RAW=1; RAWDIR="$(cd "$RAWDIR" && pwd)"; fi
if [ $REBUILD = 1 ] && [ $HAVE_RAW = 0 ]; then
    echo "--rebuild needs the raw POPANE Study 6 files at $RAWDIR (or --raw DIR). REPRODUCE.md, Raw data."
    exit 2
fi
if [ -n "$SOLVER" ]; then
    [ $HAVE_M = 1 ] || { echo "--solver needs MATLAB ($MATLAB not found)."; exit 2; }
    [ -f "$ROOT/data/tbme_input/$SOLVER.mat" ] || { echo "no input data/tbme_input/$SOLVER.mat"; exit 2; }
fi

# ------------------------------------------------------------------------------ scratch copy
case "$WORK" in ""|/|"$HOME"|"$ROOT"|"$ROOT"/*) echo "refusing to use $WORK as the scratch folder"; exit 2 ;; esac
if [ -e "$WORK" ] && [ ! -e "$WORK/.physio_reproduce" ]; then
    echo "$WORK exists and was not made by this script; choose another folder with --work."
    exit 2
fi
rm -rf "$WORK" && mkdir -p "$WORK" && WORK="$(cd "$WORK" && pwd)"
echo "Copying the project to $WORK ..."
rsync -a --exclude '/.git' --exclude '/.venv' --exclude '/data/raw' --exclude '/archive' \
      --exclude '/reproduce_report.csv' "$ROOT/" "$WORK/"
touch "$WORK/.physio_reproduce"
if [ $HAVE_RAW = 1 ]; then mkdir -p "$WORK/data/raw" && ln -s "$RAWDIR" "$WORK/data/raw/Study6"; fi
mkdir -p "$WORK/logs"; MAN="$WORK/logs/manifest.tsv"; : > "$MAN"
sleep 1; touch "$WORK/.reproduce_start"; sleep 1        # files rewritten after this count as recomputed

echo "Python: $PY   MATLAB: $([ $HAVE_M = 1 ] && command -v "$MATLAB" || echo 'not found, stage 2 skipped')"
echo "Raw POPANE files: $([ $HAVE_RAW = 1 ] && echo "$RAWDIR" || echo 'not present, steps that need them skipped')"
echo "Logs: $WORK/logs"
T0=$(date +%s)

record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$MAN"; }

run_logged() {   # run_logged NAME COMMAND...: one line of progress, the output in logs/NAME.log
    local name="$1"; shift
    local t=$(date +%s)
    printf '  %-32s' "$name"
    if (cd "${RUN_DIR:-$WORK}" && "$@") > "$WORK/logs/$name.log" 2>&1; then STATUS=OK; else STATUS=FAILED; fi
    echo "$STATUS  ($(( $(date +%s) - t )) s)"
}

py() {           # py [raw] COMMAND "OUTPUTS": PYTHONPATH=src python -m eda.run COMMAND
    local need=none
    if [ "$1" = raw ]; then need=raw; shift; fi
    local cmd="$1" outs="$2"
    if [ $need = raw ] && [ $HAVE_RAW = 0 ]; then record "$cmd" "SKIPPED: needs the raw POPANE files" "$outs"; return; fi
    run_logged "$cmd" env PYTHONPATH="$WORK/src" MPLBACKEND=Agg "$PY" -m eda.run "$cmd"
    record "$cmd" "$STATUS" "$outs"
}

m() {            # m [raw|synth] NAME 'MATLAB CALL' "OUTPUTS": src/stage2 on the path, root = the copy
    local need=none
    if [ "$1" = raw ] || [ "$1" = synth ]; then need="$1"; shift; fi
    local name="$1" call="$2" outs="$3"
    if [ $HAVE_M = 0 ]; then record "$name" "SKIPPED: MATLAB not found" "$outs"; return; fi
    if [ $need = raw ] && [ $HAVE_RAW = 0 ]; then record "$name" "SKIPPED: needs the raw POPANE files" "$outs"; return; fi
    if [ $need = synth ] && [ $FAST = 1 ]; then record "$name" "SKIPPED: --fast" "$outs"; return; fi
    run_logged "$name" "$MATLAB" -batch "addpath(fullfile(pwd, 'src', 'stage2')); root = pwd; $call"
    record "$name" "$STATUS" "$outs"
}

# ------------------------------------------------- 1. from the raw files (--rebuild only)
if [ $REBUILD = 1 ]; then
    echo "1. Rebuilding the solver inputs and the test truth from the raw files"
    py raw quality                data/quality_study6.csv
    py raw profile                "data/profile_study6.csv data/profile_study6_peaks.csv"
    py raw synth                  "data/tbme_synth_cohort.csv data/tbme_synth_check.csv"
    py raw raw-atlas              "data/raw_overview.csv results/figures/raw_atlas_*.png"
    py raw preprocess-stages      "data/preprocess_stages.csv results/figures/stages_*.png"
    py raw preprocess-stages-own4 results/figures/stages_P53_own4.png
    py raw own-check              "data/input_check.csv data/filter_response.csv results/figures/filter_response.png"
    py raw own-export             "data/tbme_synth/*_own.npz"
    py raw own4-export            "data/input_check_own4.csv data/tbme_synth/S*.npz"
    py raw noise-spectrum         "data/noise_spectrum.csv data/noise_spectrum_curves.csv"
    py raw select-measures        data/selection_measures.csv
    py select                     "data/selection.csv data/selection_balance.csv"
    py raw final-export           "data/tbme_input/P*_own4.mat data/tbme_synth/F*_own4.npz data/tbme_synth/T*_own4.npz data/tbme_test50_cohort.csv"
    py raw synth-check            "data/synth50_check.csv data/noise_spectrum50_curves.csv"
    py raw tune50-export          "data/tbme_input/U*_own4.mat data/tbme_synth/U*.npz data/tune50_cohort.csv"
    py raw test50-export          "data/tbme_input/V*_own4.mat data/tbme_synth/V*_own4.npz data/test50_cohort.csv"
    py raw own-scale              data/own_scale.csv    # reads inputs only the steps above make (U slow, F, T)
fi

# ------------------------------------- 2. stage 1 from the shipped deconvolution outputs
echo "2. Stage 1: scores, pulse tables and figures from the deconvolution outputs"
py select              "data/selection.csv data/selection_balance.csv"
py own-score           data/own_tune_scores.csv
py own-score-selected  data/own_tune_scores_selected.csv
py own-decay-check     "data/own_decay_check.csv data/own_decay_sessions.csv"
py own-errors          "data/own_errors.csv data/own_errors_summary.csv"
py round-compare       "data/round8_compare.csv results/figures/round8_tonic.pdf"
py final-score         data/own_final_test_scores.csv
py test50-score        data/own_test50_scores.csv
py resynth-export      "data/resynth50_cohort.csv data/tbme_input/R*_own4.mat data/tbme_synth/R*_own4.npz"
py resynth-score       data/own_resynth50_scores.csv
py real-pulses         "data/real50_pulses.csv data/real50_windows.csv"
py block-rates         data/real50_block_rates.csv
py extra-pulses        "data/extra47_pulses.csv data/extra47_windows.csv"
py new50-numbers       "data/new50_numbers.csv data/new50_participants.csv"
py final-plots         "results/figures/test50_deconv.pdf results/figures/real50_deconv.pdf"
py resynth-plots       results/figures/resynth50_deconv.pdf
if [ $REBUILD = 1 ]; then
    py raw scr-shapes  "data/scr_shapes_real50.csv data/prior_measured50.csv data/scr_shapes_validation.csv"
    py raw pulse-check data/pulse_check_isolated.csv
fi

# ------------------------------------------------ 3. stage 2: arousal state and H1-H6
echo "3. Stage 2: arousal state, H1-H6 and the slider (MATLAB)"
m synth s2_synth  "s2_synth(root)"  data/stage2_synth_scores.csv
m s2_run          "s2_run(root)"    "data/stage2_params.csv data/stage2_states_005.csv data/stage2_states_002.csv"
m raw s2_slider   "s2_slider(root)" data/stage2_slider_1hz.csv
m s2_tests        "s2_tests(root)"  "data/stage2_tests.csv data/stage2_block_summary.csv results/figures/stage2_hypotheses.png"
m s2_plot         "s2_plot(99, root)" results/figures/stage2_state_P99.png
m s2_plot_all     "s2_plot_all(root)" results/figures/stage2_states_real50.pdf
m s2_slider_eval  "s2_slider_eval(root)" "data/stage2_slider_eval.csv data/stage2_slider_windows.csv results/figures/stage2_slider_eval.png"
for v in tonic tonic14b; do
    m synth s2_synth_$v "s2_synth_tonic(root, '$v')" data/stage2_synth_${v}_scores.csv
    m s2_run_$v         "s2_run_tonic(root, '$v')"   "data/stage2_states_005_$v.csv data/stage2_params_$v.csv"
    m s2_tests_$v       "s2_tests(root, 'variant', '$v')" "data/stage2_tests_$v.csv data/stage2_block_summary_$v.csv results/figures/stage2_hypotheses_$v.png"
done
m synth s2_synth_tonic14d          "s2_synth_tonic(root, 'tonic14d')"         data/stage2_synth_tonic14d_scores.csv
m synth s2_synth_tonic14c          "s2_synth_tonic(root, 'tonic14c', 900000)" data/stage2_synth_tonic14c_scores.csv
m synth s2_synth_tonic14c_oldseeds "s2_synth_tonic(root, 'tonic14c', 0)"      data/stage2_synth_tonic14c_oldseeds_scores.csv
m s2_run_tonic14c         "s2_run_tonic(root, 'tonic14c')"   "data/stage2_states_005_tonic14c.csv data/stage2_params_tonic14c.csv"
m s2_tests_tonic14c       "s2_tests(root, 'variant', 'tonic14c')" "data/stage2_tests_tonic14c.csv data/stage2_block_summary_tonic14c.csv results/figures/stage2_hypotheses_tonic14c.png"
m s2_slider_eval_tonic14c "s2_slider_eval(root, 'variant', 'tonic14c')" "data/stage2_slider_eval_tonic14c.csv data/stage2_slider_windows_tonic14c.csv results/figures/stage2_slider_eval_tonic14c.png"

# ----------------------------------------- 4. one session through the deconvolution (--solver)
if [ -n "$SOLVER" ]; then
    echo "4. The deconvolution on $SOLVER (51-90 min per session on the cluster's 4 workers; longer on fewer)"
    case "$SOLVER" in
        R*) BUNDLE=make_own_resynth_bundle.sh; REF=hpc/own/resynth/output ;;
        P*|V*) BUNDLE=make_own_final_bundle.sh; REF=hpc/own/final/output
               ls "$WORK"/hpc/own/final/output/*/"${SOLVER}_events.csv" >/dev/null 2>&1 \
                   || REF=hpc/own/extra/output ;;         # one of the other 47
        *) echo "--solver takes a P, V or R session"; exit 2 ;;
    esac
    TAG=$(sed -n 's/^tag=//p' "$WORK/src/own/$BUNDLE")                  # the frozen setting, as run
    ARGS=$(sed -n 's/^args="\(.*\)"$/\1/p' "$WORK/src/own/$BUNDLE")
    SD="$WORK/solver_check"; mkdir -p "$SD/input" "$SD/output"
    cp "$WORK"/src/own/*.m "$SD/"                       # the cluster bundle's layout: own_run reads
    cp "$WORK/data/tbme_input/$SOLVER.mat" "$SD/input/" # input/ and writes output/<tag>/ next to itself
    RUN_DIR="$SD"
    run_logged "solver_$SOLVER" "$MATLAB" -batch "own_run('$SOLVER', 'amin2021', 'tag', '$TAG', $ARGS)"
    RUN_DIR="$WORK"
    outs=""
    for k in events windows signals; do outs="$outs solver_check/output/$TAG/${SOLVER}_$k.csv=$REF/$TAG/${SOLVER}_$k.csv"; done
    record "solver_$SOLVER" "$STATUS" "$outs"
fi

echo "All steps done in $(( ($(date +%s) - T0) / 60 )) min."
"$PY" "$ROOT/reproduce/compare.py" "$ROOT" "$WORK"
