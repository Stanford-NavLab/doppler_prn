#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=2:00:00
#SBATCH --job-name="gps_l1"
#SBATCH --mail-user=yalan@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=gps_l1_%j.txt
#SBATCH --error=gps_l1_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=2G
#SBATCH --partition=normal
#####################################

# run using sbatch --array=0-100:10 gps_l1.sh to run with parameters 0,2,4,6,8

# Load module for Gurobi and Julia (should be most up-to-date version, i.e. 1.7.2)
module load python/3.9

# Change to the directory of script
export SLURM_SUBMIT_DIR=/home/groups/gracegao/prn_codes/doppler_prn

# Change to the job directory
cd $SLURM_SUBMIT_DIR

lscpu

mkdir results


python3 run.py --s 0 --f 6e3 --t 9.77517107e-7 --m 31 --n 1023 --doppreg $SLURM_ARRAY_TASK_ID --maxit 10_000_000 --name "results/gps_l1" --log 1_000 --obj --obj_v_freq
