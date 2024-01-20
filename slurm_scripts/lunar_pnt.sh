#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=2:00:00
#SBATCH --job-name="lunar_pnt"
#SBATCH --mail-user=yalan@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=lunar_pnt_%j.txt
#SBATCH --error=lunar_pnt_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=2G
#SBATCH --partition=normal
#####################################

# run using sbatch --array=0,10,20,30,40,50,1000 lunar_pnt.sh

# Load module for Gurobi and Julia (should be most up-to-date version, i.e. 1.7.2)
module load python/3.9

# Change to the directory of script
export SLURM_SUBMIT_DIR=/home/groups/gracegao/prn_codes/doppler_prn

# Change to the job directory
cd $SLURM_SUBMIT_DIR

lscpu

mkdir results


python3 run.py --s 0 --f 9.5e3 --t 1.941747572815534e-7 --m 8 --n 5113 --doppreg $SLURM_ARRAY_TASK_ID --maxit 1_000_000_000 --name "results/lunar_pnt" --log 1_000 --obj --obj_v_freq
