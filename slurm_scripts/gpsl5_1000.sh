#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=48:00:00
#SBATCH --job-name="gpsl5_1000"
#SBATCH --mail-user=yalan@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=gpsl5_1000%j.txt
#SBATCH --error=gpsl5_1000%j.txt
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

python3 run.py --s 0 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 1 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 2 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 3 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 4 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 5 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 6 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 7 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 8 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
python3 run.py --s 9 --f 6e3 --t 9.77517107e-8 --m 31 --n 10230 --gs 1000 --maxit 10_000_000 --name "results/gpsl5_1000" --log 10_000 --obj
