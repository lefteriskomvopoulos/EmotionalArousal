#!/bin/bash
#SBATCH --job-name=TBME
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --array=1-2
#SBATCH --account=torch_pr_1349_general
#SBATCH --requeue
#SBATCH --output=slurm-tbme-%A_%a.out

# Course template (run_batchcontrol.sbatch); only the time, the array and the command differ.
# Array task n runs the session on line n of pids.txt (P<id> real, S<id> synthetic).
# make_bundle.sh sets --array to the number of sessions.  An optional second column on a line
# is the window length in seconds (default: 400 samples, i.e. 200 s at 2 Hz as published; -
# for the default when more columns follow).  Optional third and fourth columns fix tau
# (tau_rise tau_decay, bounds narrowed to +-5%, tests only).
# 4 random starts per window: in the local checks all 8 converged to tau within ~0.05 s.

module purge
module load matlab/2025b

export MATLAB_PREFDIR=$(mktemp -d /tmp/matlab-XXXX)
export MATLAB_LOG_DIR=/tmp
cp -rp /share/apps/matlab/shared/parallel.mlsettings ${MATLAB_PREFDIR}

read pid win tr td <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" pids.txt)"
[ "$win" = "-" ] && win=
tau=${tr:+, [$tr $td]}

cat<<MAT | matlab -nodisplay
parpool('local', 12)
tbme_run('${pid}', 4, Inf, ${win:-[]}${tau})
exit
MAT
