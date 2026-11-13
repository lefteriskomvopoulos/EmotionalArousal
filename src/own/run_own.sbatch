#!/bin/bash
#SBATCH --job-name=OWN
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --array=1-108
#SBATCH --account=torch_pr_1349_general
#SBATCH --requeue
#SBATCH --output=slurm-own-%A_%a.out

# Array task n runs line n of jobs.txt, which holds the arguments of own_run.m.
# make_own_bundle.sh writes jobs.txt and sets --array.
#
# matlab -batch, not commands piped into matlab: on an error it prints the error and exits
# non-zero, so sacct shows FAILED.  Piped commands once ended silently with COMPLETED.

echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname), $(date)"
module purge
module load matlab/2025b
echo "matlab: $(which matlab)"

export MATLAB_PREFDIR=$(mktemp -d /tmp/matlab-XXXX)
export MATLAB_LOG_DIR=/tmp
cp -rp /share/apps/matlab/shared/parallel.mlsettings ${MATLAB_PREFDIR}

job=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs.txt)
echo "own_run(${job})"

matlab -batch "parpool('local', 4); own_run(${job})"
status=$?
echo "matlab exit status ${status}, $(date)"
exit ${status}
