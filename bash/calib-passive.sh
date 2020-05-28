#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --workdir=.
#SBATCH --output=calib-passive_%j.out 
#SBATCH --error=calib-passive_%j.err
#SBATCH --job-name=calib-passive
module load gcc
module load python/3.6
python passive_props.py calib 45.89 311.89 30
