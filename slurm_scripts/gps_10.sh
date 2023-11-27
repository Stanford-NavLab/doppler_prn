#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=48:00:00
#SBATCH --job-name="gps_10"
#SBATCH --mail-user=yalan@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=gps_10o%j.txt
#SBATCH --error=gps_10e%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
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

python3 run.py --s 0 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 1 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 2 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 3 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 4 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 5 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 6 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 7 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 8 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
python3 run.py --s 9 --f 5e3 --t 9.77517107e-7 --m 31 --n 1023 --gs 10 --maxit 10_000_000 --name "results/gps_10" --log 10_000 --obj True
