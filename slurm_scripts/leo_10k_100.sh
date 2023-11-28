#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=48:00:00
#SBATCH --job-name="leo_10k_100"
#SBATCH --mail-user=yalan@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=leo_10k_100%j.txt
#SBATCH --error=leo_10k_100%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --partition=normal
#####################################

# Load module for Gurobi and Julia (should be most up-to-date version, i.e. 1.7.2)
module load python/3.9

# Change to the directory of script
export SLURM_SUBMIT_DIR=/home/groups/gracegao/prn_codes/doppler_prn

# Change to the job directory
cd $SLURM_SUBMIT_DIR

lscpu

mkdir results

python3 run.py --s 0 --f 29.6e3 --t 2e-7 --m 300 --n 10007 --gs 100 --maxit 1_000_000_000 --name "results/leo_10k_100" --log 10_000 --no-obj
