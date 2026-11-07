#!/bin/bash
#SBATCH --job-name=TBMEprobe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_1349_general
#SBATCH --output=slurm-probe-%j.out

module purge
module load matlab/2025b
export MATLAB_PREFDIR=$(mktemp -d /tmp/matlab-XXXX)
export MATLAB_LOG_DIR=/tmp

cat<<MAT | matlab -nodisplay
tbme_filter_probe
exit
MAT
