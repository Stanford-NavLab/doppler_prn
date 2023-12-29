#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=48:00:00
#SBATCH --job-name="gps_l5"
#SBATCH --mail-user=yalan@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=gps_l5%j.txt
#SBATCH --error=gps_l5%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
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

python3 run.py --s $SLURM_ARRAY_TASK_ID --f 4.5e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 0 --maxit 1_000_000_000 --name "results/gps_l5" --log 350_000 --obj --obj_v_freq
python3 run.py --s $SLURM_ARRAY_TASK_ID --f 4.5e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 0 --maxit 1_000_000_000 --name "results/gps_l5_no_doppler" --log 350_000 --obj --obj_v_freq --ignore_doppler
