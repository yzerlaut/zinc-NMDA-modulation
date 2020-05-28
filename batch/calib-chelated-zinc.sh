#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --time=02:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
# - #SBATCH --gres=gpu:1
# - #SBATCH --nodes=1
#SBATCH --chdir=.
#SBATCH --output=calib-chelated-zinc_%j.out 
#SBATCH --error=calib-chelated-zinc_%j.err
#SBATCH --job-name=calib-chelated-zinc
#SBATCH --mail-user=yann.zerlaut@icm-institute.org   # your mail
#SBATCH --mail-type=ALL # type of notifications you want to receive
module load gcc
module load python/3.6
python calibration-runs.py chelated-zinc-calib 70.0 50 20 500.0
python calibration-runs.py chelated-zinc-calib 70.0 50 20 1500.0
python calibration-runs.py chelated-zinc-calib 70.0 50 60 500.0
python calibration-runs.py chelated-zinc-calib 70.0 50 60 1500.0
python calibration-runs.py chelated-zinc-calib 70.0 80 20 500.0
python calibration-runs.py chelated-zinc-calib 70.0 80 20 1500.0
python calibration-runs.py chelated-zinc-calib 70.0 80 60 500.0
python calibration-runs.py chelated-zinc-calib 70.0 80 60 1500.0
python calibration-runs.py chelated-zinc-calib 130.0 50 20 500.0
python calibration-runs.py chelated-zinc-calib 130.0 50 20 1500.0
python calibration-runs.py chelated-zinc-calib 130.0 50 60 500.0
python calibration-runs.py chelated-zinc-calib 130.0 50 60 1500.0
python calibration-runs.py chelated-zinc-calib 130.0 80 20 500.0
python calibration-runs.py chelated-zinc-calib 130.0 80 20 1500.0
python calibration-runs.py chelated-zinc-calib 130.0 80 60 500.0
python calibration-runs.py chelated-zinc-calib 130.0 80 60 1500.0
